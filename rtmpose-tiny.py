import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mmpose.apis import MMPoseInferencer
from fastapi.responses import JSONResponse
from mmpose.utils import register_all_modules
import base64
import tempfile
import os
import urllib.request

# Register all MMPose modules (necessary step)
register_all_modules()

# Create FastAPI application instance
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, in production, it's recommended to specify domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Define local file storage paths
# Configuration file path for RTMPose-T
local_config_path = "./mmpose_configs/rtmpose-t_8xb256-420e_coco-256x192.py"
# Pre-trained model weights path for RTMPose-T
local_checkpoint_path = "./mmpose_models/rtmpose-tiny_simcc-coco_420e-256x192.pth"
# Path for the missing base runtime configuration file
local_default_runtime_path = "./_base_/default_runtime.py" # Corrected to a relative path

# Define corresponding file download URLs
config_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/dev-1.x/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py"
checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth" # RTMPose-Tiny checkpoint
default_runtime_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/dev-1.x/configs/_base_/default_runtime.py"

# Helper function: Download file if it doesn't exist
def download_if_not_exist(url, save_path):
    """
    Checks if a file exists, and if not, downloads it from the specified URL to the given path.
    """
    # Ensure the directory for saving the file exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"正在下載 {url} ...")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"已儲存到 {save_path}")
        except Exception as e:
            print(f"下載 {url} 到 {save_path} 時發生錯誤: {e}")
    else:
        print(f"{save_path} 已存在。")

# Download all necessary configuration files and model weights
download_if_not_exist(config_url, local_config_path)
download_if_not_exist(checkpoint_url, local_checkpoint_path)
download_if_not_exist(default_runtime_url, local_default_runtime_path) # Download the missing default_runtime.py

# Determine the available device (CUDA or CPU)
try:
    import torch
    if torch.cuda.is_available():
        device_str = "cuda:0"
        print("檢測到 CUDA，將使用 GPU 進行推論。")
    else:
        device_str = "cpu"
        print("未檢測到 CUDA，將使用 CPU 進行推論。")
except ImportError:
    device_str = "cpu"
    print("PyTorch 未安裝或 CUDA 不可用，將使用 CPU 進行推論。")

# Instantiate MMPoseInferencer
inferencer = MMPoseInferencer(
    pose2d=local_config_path,
    pose2d_weights=local_checkpoint_path,
    device=device_str  # Use the determined device
)

# Helper function: Cleanse numpy data types for JSON serialization
def clean_numpy(obj):
    """
    Recursively converts numpy arrays and numpy numeric types to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy(i) for i in obj]
    else:
        return obj

# Image pose estimation route
@app.post("/pose")
async def estimate_pose(file: UploadFile = File(...)):
    """
    Receives an image file, performs pose estimation, and returns keypoint coordinates and
    a base64-encoded visualization image.
    """
    # Read the uploaded image and convert it to a numpy array
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Read in BGR format

    if img_np is None:
        return {"error": "無法解碼圖片檔案。請確保上傳的是有效圖片。"}

    try:
        # Perform inference and request visualization results
        # MMPoseInferencer expects RGB images, OpenCV reads in BGR, so convert
        result_gen = inferencer(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), return_vis=True)
        result = next(result_gen) # Get the inference result
    except StopIteration:
        return {"error": "未檢測到姿態。"}
    except Exception as e:
        return {"error": f"推論期間發生錯誤: {e}"}

    # Get predicted keypoint coordinates
    predictions = result.get("predictions")
    cleaned_predictions = clean_numpy(predictions) # Clean numpy types for JSON serialization

    # Process the visualization image into Base64 encoding
    vis_img = result.get("visualization")
    vis_base64 = None
    if vis_img is not None and isinstance(vis_img, np.ndarray) and vis_img.size > 0:
        # MMPose's visualization result is in RGB format, convert back to BGR for OpenCV to encode as JPG
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        vis_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "predictions": cleaned_predictions,
        "visualization": vis_base64
    })

# Video pose estimation route
@app.post("/pose_video")
async def estimate_pose_video(file: UploadFile = File(...)):
    """
    Receives a video file, performs pose estimation frame by frame, and returns keypoint coordinates
    and a base64-encoded visualization image for each frame.
    """
    # Read the video file content
    video_bytes = await file.read()

    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        video_path = temp_video_file.name
        temp_video_file.write(video_bytes)

    # Use OpenCV to open the temporary video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        os.remove(video_path)  # Clean up temporary file
        return {"error": "無法打開影片檔案。請確保上傳的是有效影片。"}

    frames = []
    # Read each frame from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MMPoseInferencer expects RGB images, OpenCV reads in BGR, so convert
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        os.remove(video_path)  # Clean up temporary file
        return {"error": "影片中未找到任何幀。"}

    frame_results = []
    try:
        # Batch inference for all frames
        # Ensure return_vis is True to get visualization for video frames
        result_gen = inferencer(frames, return_vis=True)
        for idx, result in enumerate(result_gen):
            predictions = result.get("predictions")
            cleaned_predictions = clean_numpy(predictions)

            vis_img = result.get("visualization")
            vis_base64 = None
            if vis_img is not None and isinstance(vis_img, np.ndarray) and vis_img.size > 0:
                # MMPose's visualization result is in RGB format, convert back to BGR for OpenCV to encode as JPG
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                vis_base64 = base64.b64encode(buffer).decode("utf-8")

            frame_results.append({
                "frame_idx": idx,
                "predictions": cleaned_predictions,
                "visualization": vis_base64
            })
    except Exception as e:
        os.remove(video_path)  # Clean up temporary file
        return {"error": f"影片推論期間發生錯誤: {e}"}
    finally:
        # Ensure temporary file is deleted
        if os.path.exists(video_path):
            os.remove(video_path)

    # Return results for all frames
    return JSONResponse(content={"frames": frame_results})
