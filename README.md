# Hybrid Aerial Object Detector

A deep learning system that detects objects like vehicles, planes, ships, and buildings in drone (UAV) images. It combines two models — **Strip R-CNN** for finding object regions and a **Vision Transformer (ViT)** for understanding the full image context — to get better detection accuracy than using either model alone.

Built with **FastAPI**, **PostgreSQL**, and **Docker**. Comes with a browser-based frontend where you can upload an image and see bounding boxes drawn live on the result.

---

## What Problem This Solves

Detecting small objects in aerial images is hard because:
- Objects look different from above than from ground level
- Images are very large (1024×1024 pixels or more)
- Many objects are clustered together and need to be separated
- Classes like "small-vehicle" and "large-vehicle" look similar from above

Standard detectors like Mask R-CNN struggle here because their convolution filters are square — they treat all directions equally. Aerial objects like planes, ships, and roads are long and thin, so horizontal and vertical features matter differently. Strip convolutions fix this.

---

## How It Works — Step by Step

### Step 1: Image goes in

You upload a UAV image (JPEG, PNG, TIFF). It gets resized to 1024×1024 and normalized. The image is fed into two networks at the same time.

### Step 2: Strip FPN extracts features

The **Strip Feature Pyramid Network (FPN)** processes the image using `StripConvBlock` layers. Instead of square 3×3 filters, each block uses two separate filters:
- A **horizontal strip** filter: `1×7` kernel — captures wide objects like roads and runways
- A **vertical strip** filter: `7×1` kernel — captures tall objects like bridges and towers

These two outputs are concatenated, then passed through BatchNorm and ReLU. The FPN produces two feature maps at different scales (P3 at 1/8 resolution, P4 at 1/16) so the model can detect both small and large objects.

### Step 3: Vision Transformer encodes global context

The same image also goes into the **ViT Encoder**. The ViT splits the image into 16×16 pixel patches (4,096 patches for a 1024×1024 image), converts each patch to a 768-dimensional vector, and adds a special `[CLS]` token at the front.

These patch vectors go through 6 **Transformer blocks**. Each block has:
- **Multi-Head Self-Attention (MSA)** with 12 heads — lets every patch look at every other patch to understand relationships (e.g. the plane is near the runway)
- **Feed-forward MLP** with GELU activation
- **Layer Normalization** before each sub-layer
- **Residual connections** to prevent vanishing gradients

The output `[CLS]` token holds a 768-dim summary of the whole image. This is the global context that the FPN alone cannot see.

### Step 4: Region Proposal Network (RPN)

The FPN feature map P3 goes into the **Region Proposal Head**. It uses a 3×3 conv followed by two branches:
- **Classification branch**: scores 9 anchor boxes per location as object or background
- **Regression branch**: predicts (dx, dy, dw, dh) offsets to refine each anchor

Anchors are generated at 3 scales (32, 64, 128 px) and 3 aspect ratios (0.5, 1.0, 2.0), giving 9 anchors per location.

### Step 5: RoI Align

For each candidate region from the RPN, **RoI Align** extracts a fixed 7×7 feature map from P3 using bilinear interpolation. This avoids the quantization errors that standard RoI Pooling causes, which matters a lot for small aerial objects.

### Step 6: Hybrid Detection Head fuses both signals

This is the key part. The 7×7 RoI feature (256 channels → 12,544 values flattened) is concatenated with the ViT CLS token (768 values) to form a 13,312-dimensional vector.

This goes through:
- Linear(13312 → 1024) + ReLU + Dropout(0.3)
- Linear(1024 → 512) + ReLU

Then two output heads:
- **Classification**: Linear(512 → 16) — 15 categories + background
- **Bounding box**: Linear(512 → 64) — 4 deltas × 16 classes

Each proposal now uses both local strip features AND the full-image context.

### Step 7: Post-processing

Raw outputs go through:
- **Box decoding**: convert delta offsets back to (x1, y1, x2, y2) coordinates
- **Confidence filtering**: discard boxes below the threshold (default 0.5)
- **Non-Maximum Suppression (NMS)**: remove duplicate boxes using IoU threshold of 0.4

Final detections are returned as JSON and saved to PostgreSQL.

---

## Model Architecture Summary

```
Input Image (1024×1024×3)
         │
    ┌────┴─────────────────┐
    │                      │
 Strip FPN              ViT Encoder
 StripConvBlock×4       PatchEmbed (16×16)
 (horiz + vert filters) 6 Transformer Blocks
 → P3, P4 feature maps  → CLS token (768-dim)
    │                      │
 RPN Head                  │
 (9 anchors/location)      │
    │                      │
 RoI Align (7×7)           │
    │                      │
    └────── Concatenate ───┘
                │
       Hybrid Detection Head
       Linear 13312→1024→512
                │
        ┌───────┴───────┐
     cls head        reg head
    (16 classes)   (64 deltas)
                │
             NMS + Decode
                │
           Final Detections
```

---

## Object Categories — DOTA v2 (15 classes)

| ID | Category | ID | Category |
|----|----------|----|----------|
| 1 | plane | 9 | bridge |
| 2 | ship | 10 | large-vehicle |
| 3 | storage-tank | 11 | small-vehicle |
| 4 | baseball-diamond | 12 | helicopter |
| 5 | tennis-court | 13 | roundabout |
| 6 | basketball-court | 14 | soccer-ball-field |
| 7 | ground-track-field | 15 | swimming-pool |
| 8 | harbor | | |

---

## Results

| Model | mAP@0.5 | Inference | Notes |
|-------|---------|-----------|-------|
| Mask R-CNN baseline | 36% | ~42 FPS | Square conv filters only |
| Strip R-CNN + ViT (this project) | **48%** | **25 ms** | Strip filters + global ViT context |

The 33% relative gain comes from:
1. Strip convolutions capture elongated objects (planes, ships, roads) much better
2. The ViT CLS token provides cross-region context that local convolutions miss entirely

---

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/` | Health check |
| POST | `/detect/` | Upload image → run detection → return results + save to DB |
| GET | `/detect/stats` | Total detections, per-category counts, recent logs |

### Example response

```json
{
  "filename": "uav_sample.jpg",
  "image_width": 1024,
  "image_height": 1024,
  "total_detections": 7,
  "inference_time_ms": 24.5,
  "model_version": "v1",
  "detections": [
    {
      "category_id": 1,
      "category_name": "plane",
      "confidence": 0.91,
      "bbox": { "x1": 120.4, "y1": 88.2, "x2": 310.7, "y2": 196.5 }
    }
  ]
}
```

---

## Frontend

Open `frontend/index.html` in any browser. No install or build step needed.

Features:
- Drag and drop image upload
- Confidence and NMS threshold sliders
- Live bounding box drawing on canvas with color-coded categories
- Detection list with confidence scores
- Works in demo mode even when the API is offline (shows simulated results)

---

## Project Structure

```
aerial_detector/
├── app/
│   ├── main.py                          ← FastAPI app startup
│   ├── config.py                        ← All settings loaded from .env
│   ├── db/database.py                   ← Async PostgreSQL engine + session factory
│   ├── models/detection.py              ← SQLAlchemy tables
│   ├── schemas/detection.py             ← Pydantic request/response schemas
│   ├── services/detector_service.py     ← Load models, run inference, save to DB
│   ├── routers/detection.py             ← API route handlers
│   └── ml/
│       ├── backbone/vit_encoder.py      ← PatchEmbed + MSA + TransformerBlock + ViTEncoder
│       ├── detector/strip_rcnn.py       ← StripConvBlock + FPN + RPN + HybridDetectionHead
│       ├── utils/postprocess.py         ← Anchor generation, box decoding, NMS
│       ├── utils/transforms.py          ← Image preprocessing
│       └── train.py                     ← AdamW training loop with cosine LR
├── frontend/
│   └── index.html                       ← Upload UI with canvas bounding box rendering
├── tests/
│   └── test_detection.py
├── data/
│   ├── images/                          ← Place training images here
│   ├── checkpoints/                     ← Saved model weights (.pt files)
│   └── results/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| PyTorch | 2.3.0 | Model training and inference |
| torchvision | 0.18.0 | RoI Align, image transforms |
| einops | 0.7.0 | Tensor reshaping in ViT |
| FastAPI | 0.111.0 | Async REST API |
| SQLAlchemy | 2.0.30 | Async ORM |
| asyncpg | 0.29.0 | PostgreSQL async driver |
| Pydantic v2 | 2.7.1 | Data validation |
| PostgreSQL | 16 | Detection log storage |
| Docker | — | Containerized deployment |

---

## Setup

### Docker

```bash
git clone https://github.com/krishnaprasadlenka47/Hybrid-Aerial-Object-Detector.git
cd Hybrid-Aerial-Object-Detector
cp .env.example .env
docker-compose up --build
```

API: `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

### Local

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Training

```bash
python -m app.ml.train --epochs 12 --batch_size 4
```

Saves:
- `data/checkpoints/vit_encoder.pt`
- `data/checkpoints/detector.pt`

For full DOTA v2 training: [captain-whu.github.io/DOTA](https://captain-whu.github.io/DOTA/dataset.html)

---

## Key Design Decisions

**Why strip convolutions?**
Aerial objects like runways, roads, and ships are elongated. A `1×7` filter captures the horizontal extent of a runway far better than a `3×3` filter. Using `1×7` and `7×1` in parallel covers any orientation without rotation augmentation.

**Why concatenate ViT CLS token with RoI features?**
The RoI feature only sees the local region around a proposal. The ViT CLS token sees the entire image. Combining them lets the classifier use context like "this small box is near a harbor, so it is likely a ship" instead of deciding in isolation.

**Why RoI Align instead of RoI Pooling?**
RoI Pooling quantizes feature map coordinates to integers, which causes misalignment errors. RoI Align samples at exact floating-point positions using bilinear interpolation — critical for small objects in 1024×1024 aerial images.

**Why TensorRT INT8 for deployment?**
After training in FP32, TensorRT compresses weights to 8-bit integers, reducing memory by 4× and speeding up GPU inference by 2-4× with minimal accuracy loss. This brings latency down to 25 ms per frame.

---

## Author

**Krishna Prasad Lenka**  
Machine Learning Intern — CABS, DRDO  
[github.com/krishnaprasadlenka47](https://github.com/krishnaprasadlenka47) · [linkedin.com/in/krishna-lenka0609](https://linkedin.com/in/krishna-lenka0609) · [Portfolio](https://krishnaprasadlenka47.github.io/My_porfolio/)
