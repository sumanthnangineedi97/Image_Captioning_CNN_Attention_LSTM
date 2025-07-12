import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from src.models import EncoderCNN_LSTM, DecoderWithMLPAttention
from src.utils import build_vocab
from src.config import *
from src.inference import caption_image, load_model

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
indextoword, wordtoindex, _ = build_vocab(CAPTIONS_JSON_PATH)
vocab_size = len(wordtoindex)

# Load model
encoder = EncoderCNN_LSTM(EMBED_SIZE, HIDDEN_SIZE).to(device)
decoder = DecoderWithMLPAttention(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)
encoder, decoder = load_model(encoder, decoder, MODEL_SAVE_PATH, device)

# Load test image names
with open('./data/captions_train.json', 'r') as f:
    references = json.load(f)

# Prepare predictions
image_folder = './data/Images'
predictions = {}

for img_name in tqdm(references.keys(), desc="Generating captions"):
    image_path = os.path.join(image_folder, img_name)
    if os.path.exists(image_path):
        caption = caption_image(image_path, encoder, decoder, wordtoindex, indextoword, device)
        predictions[img_name] = caption
    else:
        print(f"⚠️ Missing image: {img_name}")

# Save predictions
with open('./data/predictions_train.json', 'w') as f:
    json.dump(predictions, f, indent=2)

print("✅ Captions saved to ./data/predictions_train.json")
