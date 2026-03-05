# Hybrid Strip R-CNN + Vision Transformer for Aerial Object Detection

## Overview

This project implements a hybrid object detection model for aerial imagery (UAV / drone-based detection).  
The architecture combines:

- Strip R-CNN (efficient region proposals for aerial scenes)
- Vision Transformer (global context modeling)

The system is deployed using:
- PyTorch
- Detectron2
- Hugging Face Transformers
- FastAPI
- PostgreSQL
- Docker
- React frontend

---

## Objectives

- Detect small objects (cars, people, buildings)
- Improve long-range dependency modeling
- Achieve mAP > 45% on DOTA
- Build full-stack deployable AI system

---

## Architecture

Input Image  
↓  
CNN Backbone  
↓  
Strip-based RPN  
↓  
Vision Transformer Encoder  
↓  
Feature Fusion  
↓  
Detection Head  

---

## Dataset

- DOTA v2.0
- xView (optional)

---

## Tech Stack

- PyTorch
- Detectron2
- Transformers
- FastAPI
- PostgreSQL
- Docker
- React

---

## Results

| Model | mAP |
|-------|------|
| Faster R-CNN Baseline | XX% |
| Hybrid StripRCNN + ViT | XX% |

---

## 🐳 Deployment

```bash
docker build -t hybrid-aerial .
docker run -p 8000:8000 hybrid-aerial
