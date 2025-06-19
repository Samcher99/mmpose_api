import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from io import BytesIO

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

if not has_mmdet:
    raise RuntimeError('Please install mmdet to run the demo.')

# Initialize FastAPI app
app = FastAPI(
    title="Pose Estimation API",
    description="API for human pose estimation using MMPose and MMDetection."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for models (load once at startup)
detector = None
pose_estimator = None
visualizer = None
args = None

# Dummy ArgumentParser for initial setup - this will be replaced by actual parameters from request or config
class DummyArgs:
    def __init__(self):
        self.det_config = ""
        self.det_checkpoint = ""
        self.pose_config = ""
        self.pose_checkpoint = ""
        self.input = ""
        self.show = False
        self.output_root = ""
        self.save_predictions = False
        self.device = "cuda:0" if os.getenv("CUDA_VISIBLE_DEVICES", None) is not None else "cpu"
        self.det_cat_id = 0
        self.bbox_thr = 0.3
        self.nms_thr = 0.3
        self.kpt_thr = 0.3
        self.draw_heatmap = False
        self.show_kpt_idx = False
        self.skeleton_style = 'mmpose'
        self.radius = 3
        self.thickness = 1
        self.show_interval = 0
        self.alpha = 0.8
        self.draw_bbox = False
        self.frame_idx = 0 # Added for FastAPI context


dummy_args = DummyArgs()

@app.on_event("startup")
async def load_models():
    global detector, pose_estimator, visualizer, args

    # Load configurations from environment variables or default paths
    dummy_args.det_config = os.getenv("DET_CONFIG", "configs/mmdet/yolox_s_8xb8-300e_coco.py")
    dummy_args.det_checkpoint = os.getenv("DET_CHECKPOINT", "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211126_233405-f93fe78d.pth")
    dummy_args.pose_config = os.getenv("POSE_CONFIG", "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py")
    dummy_args.pose_checkpoint = os.getenv("POSE_CHECKPOINT", "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/hrnet_w32_coco_256x192-c78dce93_20200708.pth")

    # If you want to use a specific GPU, set CUDA_VISIBLE_DEVICES environment variable
    # For example: export CUDA_VISIBLE_DEVICES=0

    # Build detector
    try:
        detector = init_detector(
            dummy_args.det_config, dummy_args.det_checkpoint, device=dummy_args.device
        )
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

        # Build pose estimator
        pose_estimator = init_pose_estimator(
            dummy_args.pose_config,
            dummy_args.pose_checkpoint,
            device=dummy_args.device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=dummy_args.draw_heatmap))),
        )

        # Build visualizer
        pose_estimator.cfg.visualizer.radius = dummy_args.radius
        pose_estimator.cfg.visualizer.alpha = dummy_args.alpha
        pose_estimator.cfg.visualizer.line_width = dummy_args.thickness
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=dummy_args.skeleton_style
        )
        args = dummy_args # Assign dummy_args to global args after successful loading

        print(f"Models loaded successfully on device: {dummy_args.device}")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")


def process_one_image_api(img_np: np.ndarray,
                          detector_model,
                          pose_estimator_model,
                          visualizer_instance,
                          current_args,
                          tracked_bbox: Optional[List[float]] = None):
    """
    Processes a single image for pose estimation, adapted for FastAPI.
    """
    det_result = inference_detector(detector_model, img_np)
    pred_instance = det_result.pred_instances.cpu().numpy()
    all_bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = all_bboxes[np.logical_and(pred_instance.labels == current_args.det_cat_id,
                                       pred_instance.scores > current_args.bbox_thr)]

    # Only keep the target bbox if tracked_bbox is provided
    if tracked_bbox is not None and len(bboxes) > 0:
        def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area > 0 else 0

        ious = [iou(tracked_bbox, box) for box in bboxes]
        best_idx = int(np.argmax(ious))
        bboxes = np.array([bboxes[best_idx][:4]])
    else:
        bboxes = bboxes[nms(bboxes, current_args.nms_thr), :4]

    pose_results = inference_topdown(pose_estimator_model, img_np, bboxes)
    data_samples = merge_data_samples(pose_results)

    # Convert to RGB for visualization if not already
    if isinstance(img_np, np.ndarray):
        img_np = mmcv.bgr2rgb(img_np)

    if visualizer_instance is not None:
        visualizer_instance.add_datasample(
            'result',
            img_np,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=current_args.draw_heatmap,
            draw_bbox=current_args.draw_bbox,
            show_kpt_idx=current_args.show_kpt_idx,
            skeleton_style=current_args.skeleton_style,
            show=False,
            wait_time=current_args.show_interval,
            kpt_thr=current_args.kpt_thr
        )

        if data_samples.get('pred_instances') is not None:
            # We assume a single person for simplicity here, if multiple persons are detected,
            # you might need to iterate through data_samples.pred_instances
            if len(data_samples.pred_instances) > 0:
                kpts = data_samples.pred_instances.keypoints[0]
                scores = data_samples.pred_instances.keypoint_scores[0]
                img_v = visualizer_instance.get_image()

                # Mark frame height
                frame_h = img_v.shape[0]
                cv2.putText(
                    img_v, f"frameH = {frame_h}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

                # Display frame index
                cv2.putText(
                    img_v, f"FRAME = {current_args.frame_idx:03d}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

                # Display Y values for R-Shoulder and R-Wrist
                y_info = {
                    'R-Shoulder': 6,
                    'R-Wrist': 10
                }
                for label, idx in y_info.items():
                    if scores[idx] > 0.3:
                        x, y = int(kpts[idx][0]), int(kpts[idx][1])
                        cv2.putText(
                            img_v, f"{label} Y: {y}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                        )
                visualizer_instance.set_image(img_v)
            else:
                # If no instances detected, just use the original image for visualization
                visualizer_instance.set_image(img_np)
        else:
            visualizer_instance.set_image(img_np)


    return data_samples.get('pred_instances', None)


# Response model for keypoints
class Keypoint(BaseModel):
    x: float
    y: float
    score: float

class PersonPose(BaseModel):
    bbox: List[float]
    keypoints: List[Keypoint]
    keypoint_scores: List[float]

class PoseEstimationResponse(BaseModel):
    message: str
    poses: List[PersonPose] = Field(..., description="Detected human poses with keypoints and scores.")
    frame_height: int
    frame_width: int
    # The rendered image will be sent as a separate StreamingResponse


@app.post("/pose_estimate/")
async def pose_estimate(
    file: UploadFile = File(..., description="Image file to perform pose estimation on."),
    det_cat_id: int = Form(0, description="Category ID for bounding box detection model (e.g., 0 for person in COCO)."),
    bbox_thr: float = Form(0.3, description="Bounding box score threshold."),
    nms_thr: float = Form(0.3, description="IoU threshold for bounding box NMS."),
    kpt_thr: float = Form(0.3, description="Visualizing keypoint thresholds."),
    draw_heatmap: bool = Form(False, description="Draw heatmap predicted by the model."),
    show_kpt_idx: bool = Form(False, description="Whether to show the index of keypoints."),
    skeleton_style: str = Form('mmpose', description="Skeleton style selection (mmpose or openpose)."),
    radius: int = Form(3, description="Keypoint radius for visualization."),
    thickness: int = Form(1, description="Link thickness for visualization."),
    alpha: float = Form(0.8, description="The transparency of bboxes."),
    draw_bbox: bool = Form(False, description="Draw bboxes of instances."),
    frame_idx: int = Form(0, description="Current frame index (for display purposes)."),
    tracked_bbox: Optional[str] = Form(None, description="Optional: JSON string of a bounding box [x1, y1, x2, y2] to track a specific person.")
):
    if detector is None or pose_estimator is None or visualizer is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait or check server logs.")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Update args dynamically for this request
    current_args = DummyArgs()
    current_args.det_cat_id = det_cat_id
    current_args.bbox_thr = bbox_thr
    current_args.nms_thr = nms_thr
    current_args.kpt_thr = kpt_thr
    current_args.draw_heatmap = draw_heatmap
    current_args.show_kpt_idx = show_kpt_idx
    current_args.skeleton_style = skeleton_style
    current_args.radius = radius
    current_args.thickness = thickness
    current_args.alpha = alpha
    current_args.draw_bbox = draw_bbox
    current_args.frame_idx = frame_idx
    current_args.device = dummy_args.device # Keep the device from global args

    parsed_tracked_bbox = None
    if tracked_bbox:
        try:
            parsed_tracked_bbox = json.loads(tracked_bbox)
            if not (isinstance(parsed_tracked_bbox, list) and len(parsed_tracked_bbox) == 4 and all(isinstance(x, (int, float)) for x in parsed_tracked_bbox)):
                raise ValueError("tracked_bbox must be a list of 4 numbers.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for tracked_bbox.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid tracked_bbox format: {e}")

    # Process the image
    pred_instances = process_one_image_api(
        img_np, detector, pose_estimator, visualizer, current_args, parsed_tracked_bbox
    )

    # Prepare response data
    poses_data = []
    if pred_instances is not None:
        split_results = split_instances(pred_instances)
        for instance in split_results:
            keypoints_list = []
            if 'keypoints' in instance and 'keypoint_scores' in instance:
                for kpt, score in zip(instance['keypoints'], instance['keypoint_scores']):
                    keypoints_list.append(Keypoint(x=float(kpt[0]), y=float(kpt[1]), score=float(score)))
            poses_data.append(
                PersonPose(
                    bbox=instance['bbox'].tolist(),
                    keypoints=keypoints_list,
                    keypoint_scores=instance['keypoint_scores'].tolist()
                )
            )

    # Get the visualized image
    img_vis = visualizer.get_image()
    _, im_buf_arr = cv2.imencode(".png", mmcv.rgb2bgr(img_vis))
    byte_im = BytesIO(im_buf_arr.tobytes())

    # Return JSON response first
    response_content = PoseEstimationResponse(
        message="Pose estimation successful",
        poses=poses_data,
        frame_height=img_np.shape[0],
        frame_width=img_np.shape[1]
    ).model_dump_json(indent=2) # Use model_dump_json to get a JSON string

    # To send both JSON and image, you'd typically send JSON as part of a multi-part response
    # or send the JSON, and have the client request the image from a separate endpoint.
    # For simplicity and common practice with FastAPI, let's return the image as a StreamingResponse
    # and include the pose data in the image filename or as HTTP headers if necessary.
    # A better approach for combined data would be to base64 encode the image into the JSON,
    # but for large images, this is not efficient.
    # For this example, we will send the JSON as a text response and the image as a separate endpoint.
    # Or, we can return the image and embed the keypoint data into the image metadata if needed,
    # or just return the image and the client can parse the keypoints from a separate JSON endpoint call.

    # Let's adjust to return the image and keypoints separately, or encode keypoints in JSON response.
    # For the scope of this request, I will return the JSON data, and the *rendered image* will be
    # the primary output if you want to see it visually.
    # If you want both in one go, you'd need to base64 encode the image.

    # For this response, we'll send the image as a streaming response.
    # If you need the JSON data in the same response, you would have to base64 encode the image
    # and embed it in the JSON response, or use a multipart response.
    # For direct image display, StreamingResponse is efficient.

    # Let's return the image directly and print the JSON data to console for now,
    # or you can choose to base64 encode the image and send it within the JSON.

    # Option 1: Return image as StreamingResponse (most common for image APIs)
    # The JSON data can be returned in a separate call or embedded (base64)
    # For this prompt, let's assume the user wants the image back.
    # We will print the pose data to the server logs.
    print(f"Pose data for frame {current_args.frame_idx}:\n{response_content}")
    return StreamingResponse(BytesIO(im_buf_arr.tobytes()), media_type="image/png")

# A separate endpoint to get just the pose data if needed.
# This avoids sending large base64 strings in the main image response.
@app.post("/pose_data/", response_model=PoseEstimationResponse)
async def get_pose_data(
    file: UploadFile = File(..., description="Image file to perform pose estimation on."),
    det_cat_id: int = Form(0, description="Category ID for bounding box detection model (e.g., 0 for person in COCO)."),
    bbox_thr: float = Form(0.3, description="Bounding box score threshold."),
    nms_thr: float = Form(0.3, description="IoU threshold for bounding box NMS."),
    kpt_thr: float = Form(0.3, description="Visualizing keypoint thresholds."),
    draw_heatmap: bool = Form(False, description="Draw heatmap predicted by the model."),
    show_kpt_idx: bool = Form(False, description="Whether to show the index of keypoints."),
    skeleton_style: str = Form('mmpose', description="Skeleton style selection (mmpose or openpose)."),
    radius: int = Form(3, description="Keypoint radius for visualization."),
    thickness: int = Form(1, description="Link thickness for visualization."),
    alpha: float = Form(0.8, description="The transparency of bboxes."),
    draw_bbox: bool = Form(False, description="Draw bboxes of instances."),
    frame_idx: int = Form(0, description="Current frame index (for display purposes)."),
    tracked_bbox: Optional[str] = Form(None, description="Optional: JSON string of a bounding box [x1, y1, x2, y2] to track a specific person.")
):
    if detector is None or pose_estimator is None or visualizer is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait or check server logs.")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Update args dynamically for this request
    current_args = DummyArgs()
    current_args.det_cat_id = det_cat_id
    current_args.bbox_thr = bbox_thr
    current_args.nms_thr = nms_thr
    current_args.kpt_thr = kpt_thr
    current_args.draw_heatmap = draw_heatmap
    current_args.show_kpt_idx = show_kpt_idx
    current_args.skeleton_style = skeleton_style
    current_args.radius = radius
    current_args.thickness = thickness
    current_args.alpha = alpha
    current_args.draw_bbox = draw_bbox
    current_args.frame_idx = frame_idx
    current_args.device = dummy_args.device

    parsed_tracked_bbox = None
    if tracked_bbox:
        try:
            parsed_tracked_bbox = json.loads(tracked_bbox)
            if not (isinstance(parsed_tracked_bbox, list) and len(parsed_tracked_bbox) == 4 and all(isinstance(x, (int, float)) for x in parsed_tracked_bbox)):
                raise ValueError("tracked_bbox must be a list of 4 numbers.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for tracked_bbox.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid tracked_bbox format: {e}")

    # Process the image (no visualizer needed for this endpoint as we only need data)
    det_result = inference_detector(detector, img_np)
    pred_instance = det_result.pred_instances.cpu().numpy()
    all_bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = all_bboxes[np.logical_and(pred_instance.labels == current_args.det_cat_id,
                                       pred_instance.scores > current_args.bbox_thr)]

    if parsed_tracked_bbox is not None and len(bboxes) > 0:
        def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area > 0 else 0

        ious = [iou(parsed_tracked_bbox, box) for box in bboxes]
        best_idx = int(np.argmax(ious))
        bboxes = np.array([bboxes[best_idx][:4]])
    else:
        bboxes = bboxes[nms(bboxes, current_args.nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, img_np, bboxes)
    data_samples = merge_data_samples(pose_results)

    # Prepare response data
    poses_data = []
    if data_samples.get('pred_instances') is not None:
        split_results = split_instances(data_samples.pred_instances)
        for instance in split_results:
            keypoints_list = []
            if 'keypoints' in instance and 'keypoint_scores' in instance:
                for kpt, score in zip(instance['keypoints'], instance['keypoint_scores']):
                    keypoints_list.append(Keypoint(x=float(kpt[0]), y=float(kpt[1]), score=float(score)))
            poses_data.append(
                PersonPose(
                    bbox=instance['bbox'].tolist(),
                    keypoints=keypoints_list,
                    keypoint_scores=instance['keypoint_scores'].tolist()
                )
            )

    return PoseEstimationResponse(
        message="Pose estimation data successful",
        poses=poses_data,
        frame_height=img_np.shape[0],
        frame_width=img_np.shape[1]
    )


# Function to compute pitcher score (from original script)
def compute_pitcher_score(box, frame_width, frame_height):
    x1, y1, x2, y2, score = box
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    area_score = (height * width) / (frame_width ** 2)
    center_bias = 1 - abs((xc / frame_width) - 0.5) * 2
    height_bias = yc / frame_height  # 投手通常較上方，y 小 → 加分

    return score + 0.4 * center_bias + 2.0 * area_score + 1.0 * height_bias

# You can remove or adapt `draw_topk_bboxes_with_scores` if it's only for the initial frame logic
# which is more suited for a video stream scenario rather than a single image API call.
# For this API, the `process_one_image_api` will handle the rendering.
def draw_topk_bboxes_with_scores(frame, scored_bboxes, topk=5):
    for i, (score, box) in enumerate(scored_bboxes[:topk]):
        x1, y1, x2, y2 = map(int, box[:4])
        text = f"Top{i+1} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

if __name__ == "__main__":
    import uvicorn
    # To run this application:
    # 1. Make sure you have mmdet, mmpose, mmcv, and mmengine installed.
    #    You might need to install specific versions compatible with your CUDA setup.
    #    Refer to OpenMMLab installation guides.
    #    pip install "mmcv>=2.0.0" "mmpose>=1.0.0" "mmdet>=3.0.0"
    #    pip install "mmdet>=3.0.0" "mmpose>=1.0.0"
    #    pip install "mmengine>=0.7.0"
    #    pip install fastapi uvicorn python-multipart opencv-python numpy json-tricks
    # 2. Download the pre-trained models (checkpoints and configs) if you don't have them locally.
    #    The `load_models` function tries to use environment variables or default URLs.
    #    Example for setting environment variables before running:
    #    export DET_CONFIG="path/to/your/det_config.py"
    #    export DET_CHECKPOINT="path/to/your/det_checkpoint.pth"
    #    export POSE_CONFIG="path/to/your/pose_config.py"
    #    export POSE_CHECKPOINT="path/to/your/pose_checkpoint.pth"
    #    export CUDA_VISIBLE_DEVICES="0" # To specify GPU
    # 3. Run the FastAPI application:
    #    uvicorn your_script_name:app --host 0.0.0.0 --port 8000 --reload
    #    (replace `your_script_name` with the actual name of this Python file)
    uvicorn.run(app, host="0.0.0.0", port=8000)