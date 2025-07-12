import json
import re
import os
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
def load_captions_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def build_vocab(json_path, min_freq=4):
    with open(json_path, 'r') as f:
        data = json.load(f)

    word_count = {}
    for captions in data.values():
        for caption in captions:
            for word in re.sub(r'[.!,;?]', ' ', caption.lower()).split():
                word_count[word] = word_count.get(word, 0) + 1

    tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    indextoword = {i: w for w, i in tokens}
    wordtoindex = {w: i for w, i in tokens}

    for word, count in word_count.items():
        if count > min_freq:
            idx = len(wordtoindex)
            wordtoindex[word] = idx
            indextoword[idx] = word

    return indextoword, wordtoindex, word_count



def save_best_model(loss, best_loss, encoder, decoder, enc_opt, dec_opt, epoch, path):
    if loss < best_loss:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optimizer_state_dict': enc_opt.state_dict(),
            'decoder_optimizer_state_dict': dec_opt.state_dict(),
            'loss': loss,
        }, path)
        print(f"✔️ Model saved at epoch {epoch+1} with loss {loss:.4f}")


def compute_bleu_score(model, dataset, wordtoindex, indextoword, device, num_samples=100):
    model.eval()
    smooth = SmoothingFunction().method4
    total_score = 0.0
    count = 0

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, captions = dataset[i]
            image = image.unsqueeze(0).to(device)  # (1, C, H, W)
            caption_ids = captions.tolist()

            # Ground truth text
            ref = [[indextoword[idx] for idx in caption_ids if idx not in {wordtoindex['<PAD>'], wordtoindex['<SOS>'], wordtoindex['<EOS>']}]]
            ref_tokens = [word_tokenize(" ".join(r).lower(), preserve_line=True) for r in ref]

            # Generate prediction
            pred_ids = generate_caption(model, image, wordtoindex, indextoword, device)
            pred_sentence = [indextoword[idx] for idx in pred_ids if idx not in {wordtoindex['<PAD>'], wordtoindex['<SOS>'], wordtoindex['<EOS>']}]
            pred_tokens = word_tokenize(" ".join(pred_sentence).lower(), preserve_line=True)

            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
            total_score += score
            count += 1

    return total_score / count if count > 0 else 0.0

def generate_caption(model, image, wordtoindex, indextoword, device, max_len=30):
    model.eval()
    sos_idx = wordtoindex["<SOS>"]
    eos_idx = wordtoindex["<EOS>"]

    with torch.no_grad():
        encoder_out = model.encoder(image)
        hidden_state = torch.zeros(1, 1, encoder_out.size(2)).to(device)
        cell_state = torch.zeros(1, 1, encoder_out.size(2)).to(device)

        inputs = torch.tensor([[sos_idx]]).to(device)
        output_ids = [sos_idx]

        for _ in range(max_len):
            embedded = model.decoder.embedding(inputs).squeeze(1)  # (1, word_dim)
            context, _ = model.decoder.attention(hidden_state.squeeze(0), encoder_out)
            lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (1, 1, embed+context)
            output, (hidden_state, cell_state) = model.decoder.lstm(lstm_input, (hidden_state, cell_state))
            logits = model.decoder.fc(output.squeeze(1))  # (1, vocab_size)
            predicted = logits.argmax(1).item()

            output_ids.append(predicted)
            inputs = torch.tensor([[predicted]]).to(device)

            if predicted == eos_idx:
                break

    return output_ids

