import os
import urllib.request
import cv2
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
import base64
import tempfile

# 設定 logging，使用 uvicorn 的 logger name 讓 uvicorn 可捕捉到
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger.info("[Init] Registering all MMPose modules...")
register_all_modules()

app = FastAPI()
logger.info("[Init] FastAPI instance created.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("[Init] CORS middleware applied.")

# 下載如果不存在
def download_if_not_exist(url: str, save_path: str):
    if not os.path.exists(save_path):
        logger.info(f"[Download] Downloading {url} ...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(url, save_path)
        logger.info(f"[Download] Saved to {save_path}")
    else:
        logger.info(f"[Download] File {save_path} already exists. Skipping.")

# 下載設定檔
def download_required_configs():
    base_path = "/workspace/mmpose/configs"
    config_path = os.path.join(base_path, "body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py")
    config_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    download_if_not_exist(config_url, config_path)

    base_files = {
        os.path.join(base_path, "_base_/default_runtime.py"):
            "https://raw.githubusercontent.com/open-mmlab/mmpose/master/configs/_base_/default_runtime.py",
    }
    for path, url in base_files.items():
        download_if_not_exist(url, path)

logger.info("[Init] Downloading required configs...")
download_required_configs()

local_checkpoint_path = "/workspace/mmpose/models/hrnet_w48_coco_256x192.pth"
checkpoint_url = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
logger.info("[Init] Downloading model checkpoint if needed...")
download_if_not_exist(checkpoint_url, local_checkpoint_path)

local_config_path = "/workspace/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
device = "cuda:0"
logger.info(f"[Init] Initializing model on {device} ...")
model = init_model(local_config_path, local_checkpoint_path, device=device)
logger.info("[Init] Model initialized.")

def clean_numpy(obj):
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

@app.post("/pose_video")
async def estimate_pose_video(file: UploadFile = File(...)):
    logger.info("[/pose_video] Received video for inference.")
    video_bytes = await file.read()
    logger.info("[/pose_video] Video bytes read.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        video_path = temp_video_file.name
        temp_video_file.write(video_bytes)
    logger.info(f"[/pose_video] Temporary video saved to: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("[/pose_video] Cannot open video file.")
        return {"error": "Cannot open video file"}

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    os.remove(video_path)
    logger.info(f"[/pose_video] Extracted {len(frames)} frames.")

    if not frames:
        return {"error": "No frames extracted from video."}

    frame_results = []

    for idx, frame in enumerate(frames):
        h, w = frame.shape[:2]
        bbox = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)

        results = inference_topdown(
            model,
            img=frame,
            bboxes=bbox,
            bbox_format='xyxy'
        )

        if not results:
            logger.warning(f"[/pose_video] No pose detected in frame {idx}.")
            frame_results.append({
                "frame_idx": idx,
                "keypoints": None,
                "keypoint_scores": None,
                "visualization": None
            })
            continue

        data_sample = results[0]
        keypoints = data_sample.pred_instances.keypoints
        keypoint_scores = data_sample.pred_instances.keypoint_scores

        frame_results.append({
            "frame_idx": idx,
            "keypoints": clean_numpy(keypoints),
            "keypoint_scores": clean_numpy(keypoint_scores),
            "visualization": None
        })
        logger.info(f"[/pose_video] Frame {idx} keypoints extracted.")

    logger.info("[/pose_video] Video frame-by-frame inference completed.")
    return JSONResponse(content={"frames": frame_results})