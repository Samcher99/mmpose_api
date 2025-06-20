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

啟動後，API 預設會在 [http://localhost:8000](http://localhost:8000) 提供服務。

---

## ☁️ 雲端部署（GCP Cloud Run）

使用 `Dockerfile_cloud_run` 建立專屬容器：

```bash
# Step 1: Build image (本地)
docker build -f Dockerfile_cloud_run -t gcr.io/<your-project-id>/mmpose-api .

# Step 2: Push to Google Container Registry
docker push gcr.io/<your-project-id>/mmpose-api

# Step 3: Deploy to Cloud Run
gcloud run deploy mmpose-api \
  --image gcr.io/<your-project-id>/mmpose-api \
  --platform managed \
  --region asia-east1 \
  --allow-unauthenticated
```

部署後即會自動啟動 `app.py` 中定義的 FastAPI 路由。

---
