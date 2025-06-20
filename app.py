import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mmpose.apis import MMPoseInferencer
from fastapi.responses import JSONResponse
from mmpose.utils import register_all_modules
import base64
register_all_modules()

app = FastAPI()

# 加入 CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 設定哪些來源可以存取，"*" 代表全部允許，實際部署建議改成特定網域
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法，如 GET、POST、PUT
    allow_headers=["*"],  # 允許所有 headers
)

# 實例化MMPoseInferencer
inferencer = MMPoseInferencer(
    pose2d="body",
    device="cuda:0"  # 若 Cloud Run 無 GPU，可改 'cpu'
)

# numpy不能直接傳回前端
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

@app.post("/pose")
async def estimate_pose(file: UploadFile = File(...)):
    # 把圖片轉成numpy
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        return {"error": "Cannot decode image"}

    # 丟入numpy獲得推論結果
    result_gen = inferencer(img_np,return_vis=True)
    try:
        result = next(result_gen)
    except StopIteration:
        return {"error": "No pose detected."}

    # 取得骨架座標
    predictions = result.get("predictions")
    cleaned = clean_numpy(predictions)

    # 處理 visualization 圖片為 base64
    vis_img = result.get("visualization")  # 是 RGB 格式的 np.ndarray
    
    if vis_img is None or not isinstance(vis_img, np.ndarray) or vis_img.size == 0:
        vis_base64 = None
    else:
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        vis_base64 = base64.b64encode(buffer).decode("utf-8")
        
    # 回傳JSONResponse
    return JSONResponse(content={
        "predictions": cleaned,
        "visualization": vis_base64
    })

