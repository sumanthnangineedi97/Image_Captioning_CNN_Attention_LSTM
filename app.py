from flask import Flask, request, render_template, url_for
from PIL import Image
import torch
import io
import os
import uuid
import time

from src.models import EncoderCNN, DecoderWithMLPAttention
from src.utils import build_vocab
from src.inference import caption_image_beam_search
from src.config import EMBED_SIZE, HIDDEN_SIZE, MODEL_SAVE_PATH

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load vocabulary
indextoword, wordtoindex, _ = build_vocab('./data/captions.json')
vocab_size = len(wordtoindex)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(cnn_embed_size=256).to(device)
decoder = DecoderWithMLPAttention(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                                   vocab_size=vocab_size, sos_idx=wordtoindex["<SOS>"]).to(device)

# Load trained weights
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
encoder.eval()
decoder.eval()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", caption="❗ No image provided.")

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Save image for display
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)
    image_url = url_for("static", filename=f"uploads/{filename}") + f"?t={int(time.time())}"

    # Generate caption
    caption = caption_image_beam_search(image, encoder, decoder, wordtoindex, indextoword, device)

    return render_template("index.html", caption=caption[:-6], image_url=image_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
