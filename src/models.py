import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import random
from torch.autograd import Variable
from scipy.special import expit

# Pure CNN Encoder (no LSTM)
class EncoderCNN(nn.Module):
    def __init__(self, cnn_embed_size=512):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.conv = nn.Conv2d(2048, cnn_embed_size, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Optional: fix spatial resolution

    def forward(self, images):  # (B, C, H, W)
        x = self.resnet(images)            # (B, 2048, H', W')
        x = self.adaptive_pool(x)          # (B, 2048, 7, 7)
        x = self.conv(x)                   # (B, embed_size, 7, 7)
        x = x.view(x.size(0), x.size(1), -1)      # (B, embed_size, 49)
        x = x.permute(0, 2, 1)                    # (B, 49, embed_size) → time steps for attention
        return x


# MLP-based Attention with correct shape handling
class MLPAttention(nn.Module):
    def __init__(self, encoder_dim, hidden_size):
        super(MLPAttention, self).__init__()
        self.l1 = nn.Linear(encoder_dim + hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        # encoder_outputs: (B, T, encoder_dim)
        # hidden_state: (B, hidden_size)
        B, T, H_enc = encoder_outputs.size()
        H_dec = hidden_state.size(1)

        hidden_exp = hidden_state.unsqueeze(1).repeat(1, T, 1)         # (B, T, H_dec)
        concat = torch.cat((encoder_outputs, hidden_exp), dim=2)       # (B, T, H_enc + H_dec)

        x = self.l1(concat)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        scores = self.w(x).squeeze(2)                                  # (B, T)
        attn_weights = torch.softmax(scores, dim=1)                    # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, encoder_dim)

        return context, attn_weights
    
# Decoder using the corrected MLPAttention
class DecoderWithMLPAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, sos_idx, vocab_size, word_dim=1024, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.attention = MLPAttention(encoder_dim=256, hidden_size=256)
        
        self.lstm = nn.LSTM(embed_size + word_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.sos_idx = sos_idx 
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, targets, epoch):
        B = encoder_outputs.size(0)
        T = targets.size(1)
        H = encoder_outputs.size(2)

        hidden_state = torch.zeros(1, B, H).to(encoder_outputs.device)
        cell_state = torch.zeros(1, B, H).to(encoder_outputs.device)
        inputs = torch.full((B, 1), fill_value=self.sos_idx, dtype=torch.long, device=encoder_outputs.device)

        outputs = []

        embedded_targets = self.embedding(targets)  # (B, T, word_dim)

        for t in range(T - 1):
            use_teacher_forcing = self.training and (torch.rand(1).item() < self.teacher_forcing_ratio(epoch))
            embedded_input = self.embedding(inputs.squeeze(1)) if not use_teacher_forcing else embedded_targets[:, t]

            context, _ = self.attention(hidden_state.squeeze(0), encoder_outputs)  # (B, embed_size)
            lstm_input = torch.cat([embedded_input, context], dim=1).unsqueeze(1)  # (B, 1, embed+word_dim)
            output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
            output = self.fc(output.squeeze(1))  # (B, vocab_size)
            outputs.append(output.unsqueeze(1))  # (B, 1, vocab_size)
            inputs = output.argmax(1).unsqueeze(1)

        return torch.cat(outputs, dim=1)  # (B, T-1, vocab_size)

    def teacher_forcing_ratio(self, step):
        from scipy.special import expit
        return expit(step / 20 + 0.85)


# Captioning Model
class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, epoch):
        encoder_out = self.encoder(images)        # (B, 49, embed_size)
        outputs = self.decoder(encoder_out, captions, epoch)
        return outputs
