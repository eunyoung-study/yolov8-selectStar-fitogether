# ⚽ YOLOv8 기반 FitTogether 축구장 객체 탐지

## 📌 개요
- **데이터셋**: SelectStar FitTogether 축구장 이미지 데이터셋  
- **목표**: 축구 경기 이미지에서 사람(players), 공(ball), 기타 객체(others)를 탐지하는 YOLOv8 객체 탐지 모델 학습  
- **모델**: YOLOv8 (YOLOv8n / YOLOv8s)  
- **환경**: Google Colab / Jupyter Notebook  

---

## 📂 데이터셋 구성
원본 데이터는 이미지와 JSON 형태의 annotation으로 제공되며,  
이를 **YOLO 학습 포맷**에 맞게 전처리하여 사용하였다.

```

dataset/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
├── test/
│ └── images/
└── data.yaml

```
---
## YOLO 데이터셋 `data.yaml` 생성

```python
import yaml

data = {
    "names": {
        0: "players",
        1: "ball",
        2: "others"
    }
    "path": "dataset",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
}

with open("dataset/data.yaml", "w") as f:
    yaml.dump(data, f, allow_unicode=True)

```

## 🚀 YOLOv8 모델 학습

### 📌 학습 조건
- epochs ≥ 30
- 이미지 크기(imgsz) 변경 실험 수행
- YOLOv8n / YOLOv8s 모델 사용

---

### 🔬 실험 1: 이미지 크기 비교 (YOLOv8n)

| 실험 | imgsz | epochs |
|----|----|----|
| 기본 모델 | 640 | 30 |
| 변경 모델 | 960 | 30 |

```python
from ultralytics import YOLO

# imgsz = 640
model_640 = YOLO("yolov8n.pt")
model_640.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="model_default"
)

# imgsz = 960
model_960 = YOLO("yolov8n.pt")
model_960.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=960,
    batch=8,
    name="model_imgsize_960"
)
```
### 🔬 실험 2: 모델 크기 비교 (YOLOv8n vs YOLOv8s)

- 동일한 데이터셋 사용
- 동일한 epoch 조건에서 모델 크기에 따른 성능 비교 수행

```python
from ultralytics import YOLO

model_n = YOLO("yolov8n.pt")
model_s = YOLO("yolov8s.pt")
```
---

## 📊 성능 평가 및 비교
### 평가 지표
- mAP@0.5
- mAP@0.5:0.95
- Inference Time (추론 속도)
```python
metrics = model.val(split="val")
print(metrics.box.map50, metrics.box.map)
```
📌 실험 결과,
이미지 크기(imgsz)를 증가시킬수록 탐지 성능(mAP)은 향상되었으나
추론 속도는 상대적으로 감소하는 경향을 확인하였다.

---

## 🖼️ 추론 결과 시각화
- 테스트 이미지에 대해 모델별 추론 결과 비교
- 동일 이미지에서 imgsz=640 vs imgsz=960 결과 시각화
```python
plt.subplot(1, 2, 1)
plt.imshow(result_640)
plt.title("YOLOv8n imgsz=640")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result_960)
plt.title("YOLOv8n imgsz=960")
plt.axis("off")
```
## 🧠 결론
- YOLOv8n은 빠른 추론 속도를 제공하여 실시간 응용에 적합함
- 이미지 크기(imgsz) 증가 시 작은 객체 탐지 성능이 개선됨
- 성능 향상과 추론 속도 간 trade-off를 고려한 모델 선택이 중요함
