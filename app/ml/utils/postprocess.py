import torch
import torch.nn.functional as F
from torchvision.ops import nms, box_iou
from app.config import settings


def generate_anchors(feat_h: int, feat_w: int, stride: int = 8) -> torch.Tensor:
    scales = [32, 64, 128]
    ratios = [0.5, 1.0, 2.0]
    anchors = []
    for y in range(feat_h):
        for x in range(feat_w):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for scale in scales:
                for ratio in ratios:
                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)
                    anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = torch.exp(dw.clamp(max=4.0)) * anchor_w
    pred_h = torch.exp(dh.clamp(max=4.0)) * anchor_h

    boxes = torch.stack([
        pred_cx - pred_w / 2,
        pred_cy - pred_h / 2,
        pred_cx + pred_w / 2,
        pred_cy + pred_h / 2,
    ], dim=1)
    return boxes


def postprocess_detections(
    cls_logits: torch.Tensor,
    bbox_deltas: torch.Tensor,
    anchors: torch.Tensor,
    image_size: int,
    conf_threshold: float = None,
    iou_threshold: float = None,
) -> list[dict]:
    conf_threshold = conf_threshold or settings.confidence_threshold
    iou_threshold = iou_threshold or settings.nms_iou_threshold

    probs = F.softmax(cls_logits, dim=-1)
    scores, labels = probs[:, 1:].max(dim=-1)
    labels = labels + 1

    boxes = decode_boxes(anchors[:len(cls_logits)], bbox_deltas[:, :4])

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size)

    keep_mask = scores >= conf_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    if len(boxes) == 0:
        return []

    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    results = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        results.append({
            "bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
            "confidence": round(score, 4),
            "category_id": int(label),
        })
    return results
