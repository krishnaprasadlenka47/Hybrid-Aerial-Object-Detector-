import torch
from torchvision import transforms
from PIL import Image
import io

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_inference_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_train_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_image_tensor(image_bytes: bytes, image_size: int) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    transform = get_inference_transform(image_size)
    tensor = transform(image).unsqueeze(0)
    return tensor, original_size
