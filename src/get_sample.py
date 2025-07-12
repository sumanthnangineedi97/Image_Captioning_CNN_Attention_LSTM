import argparse
import torch
from torchvision import transforms
from src.data import ImageCaptionDataset
from src.utils import load_captions_json, build_vocab
from src.config import IMAGE_DIR, CAPTIONS_JSON_PATH, TRAIN_JSON_PATH

from PIL import Image

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True, help="Index to retrieve from dataset")
args = parser.parse_args()

# Load vocab and labels
indextoword, wordtoindex, _ = build_vocab(CAPTIONS_JSON_PATH)
label_dict = load_captions_json(TRAIN_JSON_PATH)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = ImageCaptionDataset(label_dict, IMAGE_DIR, wordtoindex, transform=transform)

# Retrieve sample
image_tensor, caption_tensor = dataset[args.idx]

# Convert caption tensor to tokens
caption = [indextoword.get(idx.item(), "<UNK>") for idx in caption_tensor if idx.item() != wordtoindex["<PAD>"]]
print(f"\n🖼️ Image Tensor Shape: {image_tensor.shape}")
print("📝 Caption:", " ".join(caption))
