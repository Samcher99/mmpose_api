import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mmpose.apis import MMPoseInferencer
from fastapi.responses import JSONResponse
from mmpose.utils import register_all_modules
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

inferencer = MMPoseInferencer(
    pose2d="body",
    device="cuda:0"  # 若 Cloud Run 無 GPU，可改 'cpu'
)

'''
@app.post("/pose")
async def estimate_pose(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        return {"error": "Cannot decode image"}
    results = inferencer(img_np)
    return {"result": str(results)}
#成功了回傳這個東西{"result":"<generator object MMPoseInferencer.__call__ at 0x7b2832f1ff40>"}
'''

'''
@app.post("/pose")
async def estimate_pose(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        return {"error": "Cannot decode image"}

    # 取出 inferencer 的第一個結果
    result_gen = inferencer(img_np)
    try:
        result = next(result_gen)
    except StopIteration:
        return {"error": "No pose detected."}

    print(result)
    #這裡result是這個
    defaultdict(<class 'list'>, {'visualization': None, 'predictions': [[{'keypoints': [[693.7362060546875, 156.47781372070312], [683.406494140625, 140.98316955566406], [667.911865234375, 146.14805603027344], [608.5157470703125, 166.8075714111328], [624.0103759765625, 174.5548858642578], [546.5372314453125, 264.94024658203125], [711.8132934570312, 277.8524475097656], [435.4923095703125, 254.61050415039062], [701.4835815429688, 394.0622253417969], [399.3381652832031, 125.48854064941406], [732.4728393554688, 443.1285705566406], [567.1967163085938, 543.8436889648438], [613.6806030273438, 541.26123046875], [463.899169921875, 678.1305541992188], [778.9567260742188, 631.6466064453125], [285.7108459472656, 753.021240234375], [840.9352416992188, 838.2417602539062]], 'keypoint_scores': [0.7784907817840576, 0.7048039436340332, 0.7633109092712402, 0.4956587255001068, 0.6797255277633667, 0.5693095326423645, 0.5640244483947754, 0.6629411578178406, 0.5662106275558472, 0.801803469657898, 0.487191379070282, 0.6105688810348511, 0.6073668003082275, 0.6363389492034912, 0.739112138748169, 0.775202751159668, 0.819401741027832], 'bbox': ([162.78671264648438, 41.696266174316406, 956.112060546875, 906.5393676757812],), 'bbox_score': 0.8742195}, {'keypoints': [[937.601806640625, 10.480451583862305], [949.3655395507812, 6.202718734741211], [931.1851806640625, 7.272151947021484], [962.19873046875, -54.754974365234375], [911.9353637695312, -54.754974365234375], [991.0734252929688, 7.272151947021484], [877.7135009765625, 8.341585159301758], [1019.9481201171875, 65.02154541015625], [861.6719970703125, 71.43814086914062], [975.0319213867188, 57.53551483154297], [921.560302734375, 58.60494613647461], [966.4765014648438, 148.4373321533203], [904.4493408203125, 149.50677490234375], [973.9625244140625, 279.9776306152344], [900.171630859375, 282.1164855957031], [987.8651733398438, 401.89300537109375], [893.7550048828125, 399.754150390625]], 'keypoint_scores': [0.3410091996192932, 0.26588645577430725, 0.2760627269744873, 0.28639763593673706, 0.3041340112686157, 0.6493935585021973, 0.6442084312438965, 0.7401038408279419, 0.7697579860687256, 0.6173242330551147, 0.656730055809021, 0.6158269643783569, 0.6069598197937012, 0.6641103029251099, 0.6609752178192139, 0.8109627962112427, 0.8171224594116211], 'bbox': ([843.8890991210938, 0.0, 1039.869873046875, 438.03985595703125],), 'bbox_score': 0.77363694}]]})
'''

'''
@app.post("/pose")
async def estimate_pose(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        return {"error": "Cannot decode image"}

    result_gen = inferencer(img_np)
    try:
        result = next(result_gen)
    except StopIteration:
        return {"error": "No pose detected."}
    print(result.get("predictions"))
    return {"predictions": result.get("predictions")}
大概只能搞到這邊只能用str(result.get("predictions"))回傳 若直接回傳result.get("predictions")會報錯
ValueError: [TypeError("'numpy.float32' object is not iterable"), TypeError('vars() argument must have __dict__ attribute')]
'''

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
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        return {"error": "Cannot decode image"}

    result_gen = inferencer(img_np)
    try:
        result = next(result_gen)
    except StopIteration:
        return {"error": "No pose detected."}

    predictions = result.get("predictions")
    cleaned = clean_numpy(predictions)

    return JSONResponse(content={"predictions": cleaned})