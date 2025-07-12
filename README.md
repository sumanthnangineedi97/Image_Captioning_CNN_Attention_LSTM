## 🧠 Image Captioning using CNN Encoder & MLP-Based Attention Decoder
This repository contains the complete codebase and configuration used to train a CNN-based Encoder and MLP-Attention Decoder model for Image Captioning. The model is trained from scratch on the Flickr8k dataset using Slurm batch jobs on the Clemson Palmetto HPC cluster. The trained model is integrated into a Flask web application, containerized with Docker, and deployed on Google Cloud for live inference.

## 🎥 Live Demo

Curious how the app works in action? Check out the demo video below showcasing the image captioning web application running inside a Docker container:
[![Watch Demo](https://user-images.githubusercontent.com/placeholder/demo-thumb.png)](https://github.com/sumanthnangineedi97/Image_Captioning_CNN_Attention_LSTM/assets/f98ab1c0-b838-4bb8-bbf5-e8b0179602a6)

## 📊 Dataset

The model was trained on the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which contains 8,000 images and five human-annotated captions per image.

## Project Structure
```
.
├── app.py
├── data
│   ├── captions.json
│   ├── captions_test.json
│   ├── captions_train.json
│   ├── captions.txt
│   ├── Images
│   ├── predictions_test.json
│   └── predictions_train.json
├── Dockerfile
├── model_logs
│   └── output_Image_cap_attent_training_3953339_Image_captioning_3953339.log
├── models
│   └── best_model_attention.pth
├── requirements.txt
├── runs
│   └── image_captioning
├── src
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── generate_captions_json.py
│   ├── generate_pred_captions_json.py
│   ├── get_sample.py
│   ├── inference.py
│   ├── __init__.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── static
│   ├── style.css
│   └── uploads
├── templates
│   └── index.html
└── test
    └── test.jpg
```
## 🧠 Model Architecture

- **Encoder**: ResNet50 pretrained CNN with spatial feature map output.
- **Attention**: Multi-layer Perceptron (MLP)-based attention mechanism over spatial features.
- **Decoder**: LSTM that uses attended context + word embeddings to generate tokens.

## 🏋️ Training Instructions
This section explains how to train the CNN + MLP Attention-based Image Captioning model from scratch using the Flickr8k dataset on Clemson Palmetto (HPC) with SLURM.

🔧 Step 1: Environment Setup
    Ensure the following Python libraries are installed (can be managed via conda or pip)
```
pip install -r requirements.txt
```

🔧 Step 2: Preprocess Captions
To generate vocabulary and tokenize captions:
```
python src/data.py
```
This will create tokenized captions and build the vocabulary dictionary for training.

🔧 Step 3: Start Training
Use this command to train locally:
```
python src/train.py
```
Or submit it as a SLURM job on Palmetto:
```
sbatch script_IC.sh
```

## 🧪 Training & Evaluation Summary

- **Training Script**: [`src/train.py`](src/train.py)  
- **SLURM Job Script**: [`script_IC.sh`](train.sh)  
- **Training Logs**: Saved in the [`model_logs/`](model_logs/) directory for monitoring and debugging.  
- **Evaluation**: Model performance was evaluated using BLEU scores (BLEU-1 to BLEU-4).  
- **TensorBoard Logs**: Stored in the [`runs/image_captioning/`](runs/image_captioning/) directory for visualization and progress tracking..
## 🌐 Deployment

```bash
# Build the Docker image
docker build -t image-captioning-app .
```
```bash
# Run the Docker container
docker run --name image-captioning-container -p 8080:5000 image-captioning-app
```

