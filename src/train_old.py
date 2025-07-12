import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import ImageCaptionDataset, MyCollate
from src.models import EncoderCNN, DecoderRNN
from src.utils import load_captions_json, build_vocab, save_best_model
from src.config import *

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    label_dict = load_captions_json(TRAIN_JSON_PATH)
    indextoword, wordtoindex, _ = build_vocab(CAPTIONS_JSON_PATH)
    pad_idx = wordtoindex["<PAD>"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    dataset = ImageCaptionDataset(label_dict, IMAGE_DIR, wordtoindex, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MyCollate(pad_idx))

    # Initialize model
    encoder = EncoderCNN(EMBED_SIZE).to(device)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(wordtoindex)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    encoder.train()
    decoder.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)

            outputs = outputs[:, :-1, :].reshape(-1, outputs.size(2))
            targets = captions[:, 1:].reshape(-1)

            loss = criterion(outputs, targets)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            #progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        save_best_model(avg_loss, best_loss, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, MODEL_SAVE_PATH)
        best_loss = min(best_loss, avg_loss)

if __name__ == "__main__":
    train()
