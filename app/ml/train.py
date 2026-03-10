import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from app.ml.backbone.vit_encoder import ViTEncoder
from app.ml.detector.strip_rcnn import HybridAerialDetector
from app.ml.utils.postprocess import generate_anchors
from app.config import settings


class AerialDataset(Dataset):
    def __init__(self, image_dir: str, transform):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.image_paths[idx]


def train(epochs: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(settings.checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = AerialDataset(settings.image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    vit = ViTEncoder(
        img_size=settings.image_size,
        patch_size=16,
        embed_dim=768,
        depth=6,
        num_heads=12,
    ).to(device)

    detector = HybridAerialDetector(
        num_classes=settings.num_classes,
        vit_embed_dim=768,
        base_channels=64,
    ).to(device)

    params = list(vit.parameters()) + list(detector.parameters())
    optimizer = optim.AdamW(params, lr=settings.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cls_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        vit.train()
        detector.train()
        total_loss = 0.0
        start = time.time()

        for images, _ in loader:
            images = images.to(device)

            vit_features = vit(images)
            outputs = detector(images, vit_features)

            B, _, H, W = outputs["fpn_p3"].shape
            feat_h, feat_w = H, W
            dummy_cls_targets = torch.zeros(B, dtype=torch.long, device=device)
            loss = cls_criterion(outputs["cls_logits"][:B], dummy_cls_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch [{epoch}/{epochs}]  Loss: {total_loss/len(loader):.4f}  Time: {elapsed:.1f}s")

    torch.save(vit.state_dict(), os.path.join(settings.checkpoint_dir, "vit_encoder.pt"))
    torch.save(detector.state_dict(), os.path.join(settings.checkpoint_dir, "detector.pt"))
    print("Training complete. Checkpoints saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch_size", type=int, default=settings.batch_size)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)
