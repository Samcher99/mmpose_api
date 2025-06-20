# 🏃‍♂️ MMPose API

本專案提供基於 [MMPose](https://github.com/open-mmlab/mmpose) 的骨架姿勢預測推論服務，使用 FastAPI 架設 RESTful API，支援影片輸入與骨架分析，並支援本地 GPU 加速與 GCP 雲端部署。

---

## 📦 專案目錄概述

```
├── Dockerfile                 # 本地開發用 Docker 建構檔
├── Dockerfile_cloud_run       # 部署至 GCP Cloud Run 的版本
├── docker-compose.dev.yml     # 本地開發用 docker-compose 設定
├── app.py                     # FastAPI 主應用程式入口
└── README.md
```

---

## 🖥️ 本地開發（使用 GPU）

使用 `Dockerfile` 搭配 `docker-compose.dev.yml`：

```bash
docker-compose -f docker-compose.dev.yml up --build
```

```bash
啟動後，docker-compose.dev.yml會開啟一個bash環境 

此環境開啟一個8000 port連接到本機 並將本機當前目錄對應到docker環境的/workspace

這時可以在本機修改app.py docker中的workspace/app.py會跟著改變

修改完畢在docker中下指令uvicorn app:app --host 0.0.0.0 --port 8000

會將api服務設置在port 8000對外連出來即可進行測試

可以在此開發app.py 並進行測試。
```
---

## ☁️ 雲端部署（GCP Cloud Run）

使用 連結存放區方式部屬 選擇`Dockerfile_cloud_run` 建立專屬容器：

```bash
Step 1: 進入google cloud run控制台

Step 2: 點擊上方 服務 連結存放區

Step 3: 從存放區持續部署 (原始碼或函式) 連結此專案網址

Step 4: 設定cloud run bulid

Step 5: 建構類型 Dockerfile

Step 6: 選擇正確 Dockerfile位置 此專案選擇 Dockerfile_cloud_run
```

部署後即會自動啟動 `app.py` 中定義的 FastAPI 路由。

---
