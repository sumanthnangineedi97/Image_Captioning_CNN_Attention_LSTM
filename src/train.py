import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data import ImageCaptionDataset, MyCollate
from src.models import EncoderCNN, DecoderWithMLPAttention, CaptionModel

from src.utils import load_captions_json, build_vocab, save_best_model, compute_bleu_score
from src.config import *

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir="runs/image_captioning")

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

    encoder = EncoderCNN(cnn_embed_size=256)
    sos_idx = wordtoindex["<SOS>"]
    decoder = DecoderWithMLPAttention(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                                      vocab_size=len(wordtoindex), sos_idx=sos_idx)
    model = CaptionModel(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler()

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images, captions, epoch)  # (B, T-1, vocab_size)
                outputs = outputs.reshape(-1, outputs.size(2)).float() 
                targets = captions[:, 1:].reshape(-1)  # (B*T)

                if torch.isnan(outputs).any():
                    print("NaN detected in model outputs. Sample logits:", outputs[0])
                    continue

                loss = criterion(outputs, targets)

                if torch.isnan(loss):
                    print("NaN detected in loss! Skipping this batch.")
                    continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        save_best_model(avg_loss, best_loss, model.encoder, model.decoder, optimizer, optimizer, epoch, MODEL_SAVE_PATH)
        best_loss = min(best_loss, avg_loss)

        model.eval()
        bleu_score = compute_bleu_score(model, dataset, wordtoindex, indextoword, device)
        writer.add_scalar("BLEU/val", bleu_score, epoch)

        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, BLEU: {bleu_score:.4f}")

    writer.close()

if __name__ == "__main__":
    train()
