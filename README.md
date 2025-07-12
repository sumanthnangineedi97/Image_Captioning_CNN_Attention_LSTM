# Image Captioning using MLP-based Attention
This repo contains the code and data used to train CNNEncoder and MLP-based attention decoder model used for Image Captioning. The Model is trained from scratch on Clemson Palmetoo using the above code on FLicker datset by using slurm batch jobs. And the trained model is integrated with flask for web deploymentand dockerized and deployed on Google Cloud 

## Dataset used for training
https://www.kaggle.com/datasets/adityajn105/flickr8k

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

## 🏋️ Training Info

I have used the **Clemson Palmetto High Performance Computing (HPC) Cluster** for training the image captioning model. The training job was submitted using a SLURM batch script that requested GPU nodes with the following configuration:

- **Job Scheduler**: SLURM
- **GPU**: 2 × NVIDIA P100
- **CPU**: 16 cores
- **Memory**: 16 GB
- **Training Time**: Up to 20 hours
- **Batch Size**: 16
- **Embedding Size**: 256
- **Hidden Size**: 256
- **Learning Rate**: 1e-4
- **Epochs**: 100

The training script used: `src/train.py`  
The SLURM job script: `train.sh`  
Logs were saved in the `model_logs/` directory for monitoring and debugging.

The model's performance was evaluated using BLEU score and logged via **TensorBoard**, which is stored in the `runs/image_captioning/` directory.
## 🌐 Deployment

- Web app built using **Flask**.
- Dockerized using `Dockerfile`.
- Run locally with:
  ```bash
  docker build -t image-captioning-app .
  docker run --name my-container -p 8080:5000 image-captioning-app
  ```
  
## 🛠️ Technologies Used

- Python, PyTorch, Torchvision
- Flask
- Docker
- SLURM + HPC (Palmetto)
- TensorBoard
