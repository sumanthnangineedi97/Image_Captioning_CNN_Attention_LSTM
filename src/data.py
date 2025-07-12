import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class ImageCaptionDataset(Dataset):
    def __init__(self, label_dict, image_dir, wordtoindex, transform=None, max_len=30):
        self.label_dict = label_dict
        self.image_dir = image_dir
        self.wordtoindex = wordtoindex
        self.transform = transform
        self.max_len = max_len
        self.data_pairs = self._make_data_pairs()

    def _make_data_pairs(self):
        pairs = []
        for img, captions in self.label_dict.items():
            path = os.path.join(self.image_dir, img)
            if os.path.exists(path):
                for cap in captions:
                    pairs.append((img, cap))
        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_name, caption = self.data_pairs[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = ["<SOS>"] + caption.lower().split() + ["<EOS>"]
        caption_ids = [self.wordtoindex.get(w, self.wordtoindex["<UNK>"]) for w in tokens]
        caption_ids = caption_ids[:self.max_len] + [self.wordtoindex["<PAD>"]] * (self.max_len - len(caption_ids))

        return image, torch.tensor(caption_ids)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([x[0] for x in batch])
        captions = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=self.pad_idx)
        return images, captions
