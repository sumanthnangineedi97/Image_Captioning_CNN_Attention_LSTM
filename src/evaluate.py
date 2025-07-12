import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Use custom nltk data path
nltk.data.path.append('/home/anangin/nltk_data')

# Load ground-truth references
with open('./data/captions_train.json', 'r') as f:
    references = json.load(f)

# Load model-generated predictions
with open('./data/predictions_train.json', 'r') as f:
    predictions = json.load(f)

def evaluate_metrics(predictions, references):
    bleu_scores = []
    meteor_scores = []
    smooth = SmoothingFunction().method4

    for img_id in predictions:
        pred = predictions[img_id]
        ref_list = references[img_id]

        # Tokenize for BLEU
        ref_tokens = [word_tokenize(ref.lower(), preserve_line=True) for ref in ref_list]
        pred_tokens = word_tokenize(pred.lower(), preserve_line=True)

        # BLEU
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)

        # Tokenize for METEOR (expects tokenized inputs)
        pred_tokens_meteor = pred_tokens
        meteor = max([meteor_score([word_tokenize(ref.lower(), preserve_line=True)], pred_tokens_meteor)
                      for ref in ref_list])
        meteor_scores.append(meteor)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return avg_bleu, avg_meteor

# Run and print
avg_bleu, avg_meteor = evaluate_metrics(predictions, references)
print('sample_prections',predictions)
print(f"\n Evaluation Results")
print(f" Average BLEU Score  : {avg_bleu:.4f}")
print(f" Average METEOR Score: {avg_meteor:.4f}")
