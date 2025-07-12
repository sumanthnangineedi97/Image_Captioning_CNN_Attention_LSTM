import os
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

from src.models import EncoderCNN, DecoderWithMLPAttention
from src.utils import build_vocab
from src.config import *

def load_model(encoder, decoder, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def caption_image_beam_search(image, encoder, decoder, wordtoindex, indextoword, device, beam_size=3, max_len=30):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_out = encoder(image)  # (1, T, 256)
        T, H = encoder_out.size(1), encoder_out.size(2)
        encoder_out = encoder_out.expand(beam_size, T, H)  # (beam, T, 256)

        k = beam_size
        vocab_size = len(wordtoindex)
        sequences = [[wordtoindex["<SOS>"]]] * k
        scores = torch.zeros(k, 1).to(device)

        hidden = torch.zeros(1, k, HIDDEN_SIZE).to(device)
        cell = torch.zeros(1, k, HIDDEN_SIZE).to(device)

        complete_seqs = []
        complete_scores = []

        for _ in range(max_len):
            temp_seqs, temp_scores, temp_hidden, temp_cell = [], [], [], []

            for i, seq in enumerate(sequences):
                if seq[-1] == wordtoindex["<EOS>"]:
                    complete_seqs.append(seq)
                    complete_scores.append(scores[i])
                    continue

                input_word = torch.tensor([seq[-1]]).to(device)
                embedded = decoder.embedding(input_word).view(1, -1)  # (1, word_dim)

                context, _ = decoder.attention(hidden[0, i].unsqueeze(0), encoder_out[i].unsqueeze(0))  # (1, 256)
                lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (1, 1, word_dim+256)

                output, (h, c) = decoder.lstm(lstm_input, (hidden[:, i:i+1], cell[:, i:i+1]))
                logits = decoder.fc(output.squeeze(1))  # (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=1)

                topk_log_probs, topk_ids = log_probs.topk(k)

                for j in range(k):
                    next_seq = seq + [topk_ids[0][j].item()]
                    next_score = scores[i] + topk_log_probs[0][j]
                    temp_seqs.append(next_seq)
                    temp_scores.append(next_score)
                    temp_hidden.append(h)
                    temp_cell.append(c)

            if not temp_seqs:
                break

            scores_tensor = torch.stack(temp_scores).squeeze(1)
            topk_scores, topk_indices = scores_tensor.topk(k)

            sequences = [temp_seqs[i] for i in topk_indices]
            scores = topk_scores.unsqueeze(1)
            hidden = torch.cat([temp_hidden[i] for i in topk_indices], dim=1)
            cell = torch.cat([temp_cell[i] for i in topk_indices], dim=1)

        if complete_seqs:
            best_idx = complete_scores.index(max(complete_scores))
            best_seq = complete_seqs[best_idx]
        else:
            best_seq = sequences[0]

        caption = [indextoword.get(idx, "<UNK>") for idx in best_seq]
        return " ".join(caption[1:-1])  # remove <SOS> and <EOS>

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for decoding")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indextoword, wordtoindex, _ = build_vocab(CAPTIONS_JSON_PATH)
    vocab_size = len(wordtoindex)

    encoder = EncoderCNN(cnn_embed_size=EMBED_SIZE).to(device)
    decoder = DecoderWithMLPAttention(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=vocab_size, sos_idx=wordtoindex["<SOS>"]).to(device)

    encoder, decoder = load_model(encoder, decoder, MODEL_SAVE_PATH, device)

    image = Image.open(args.image).convert("RGB")
    caption = caption_image_beam_search(image, encoder, decoder, wordtoindex, indextoword, device, args.beam_size)
    print(f"\n🖼️ Caption: {caption}")
