import os
import re
from pathlib import Path
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import tqdm

import src.open_clip as open_clip

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_label_mappings():
    """
    Define text descriptions and label indices for osteosarcoma classification.
    """
    cats_dict = {
        "Non-tumor": "Non-tumor",
        "Necrotic tumor": "Necrotic tumor",
        "Viable tumor": "Viable tumor"
    }

    label_dict = {
        "Non-tumor": 0,
        "Necrotic tumor": 1,
        "Viable tumor": 2
    }

    csv2label = {
        "Non-Tumor": "Non-tumor",
        "Viable": "Viable tumor",
        "viable: non-viable": "Viable tumor",
        "Non-Viable-Tumor": "Necrotic tumor"
    }

    return cats_dict, label_dict, csv2label


def parse_csv(csv_path: Path, image_root: Path, csv2label: dict):
    """
    Parse CSV to construct a list of image paths and corresponding ground-truth labels.
    """
    df = pd.read_csv(csv_path)
    img_list = []
    img2label = {}

    for _, row in df.iterrows():
        raw_name = row['image.name']
        norm_name = re.sub(' +', '-', raw_name)
        norm_name = re.sub('-+', '-', norm_name) + '.jpg'

        label_name = csv2label[row['classification']]
        full_path = str(image_root / norm_name)

        img_list.append(full_path)
        img2label[full_path] = label_name

    return img_list, img2label


class PathologyDataset(Dataset):
    """Standard pathology dataset that maps image paths to preprocessed tensors and label indices."""
    def __init__(self, img_list, img2label, preprocess, label_dict):
        self.img_list = img_list
        self.img2label = img2label
        self.preprocess = preprocess
        self.label_dict = label_dict

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")
        label_name = self.img2label[img_path]
        label = self.label_dict[label_name]
        return self.preprocess(image), label


def prepare_text_features(cats_dict, tokenizer_path):
    """Tokenize and encode textual label descriptions."""
    label_texts = [
        f"An H&E image patch of {desc.lower()}" for desc in cats_dict.values()
    ]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(label_texts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features, cats_dict):
    """Run model inference and classification report."""
    correct = 0
    total = 0
    gts, preds = [], []

    model.eval()
    for images, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
        images, labels = images.cuda(), labels.cuda()
        total += labels.size(0)

        with torch.no_grad():
            image_feats = model.encode_image(images)
            text_feats = model.encode_text(text_features)

            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_feats @ text_feats.T).softmax(dim=-1)
            pred_labels = torch.argmax(logits, dim=-1)

        correct += (pred_labels == labels).sum().item()
        gts.extend(labels.cpu().numpy())
        preds.extend(pred_labels.cpu().numpy())

        if total % 50 == 0:
            print(f"Running accuracy: {correct / total:.4f}")

    acc = correct / total
    print(classification_report(gts, preds, target_names=list(cats_dict.values()), digits=3))
    print(f"Overall Accuracy: {acc:.4f}")


def main():
    # ===== Path configs (anonymized) =====
    csv_path = Path('pathto/osteo/ML_Features_1144.csv')
    image_root = Path('pathto/osteo/images')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 128

    # ===== Label mapping and image list =====
    cats_dict, label_dict, csv2label = load_label_mappings()
    img_list, img2label = parse_csv(csv_path, image_root, csv2label)

    # ===== Load model and tokenizer =====
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_path, cache_dir=cache_dir
    )
    model = model.cuda()
    text_features = prepare_text_features(cats_dict, tokenizer_path)

    # ===== Data and evaluation =====
    dataset = PathologyDataset(img_list, img2label, preprocess, label_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    evaluate(model, dataloader, text_features, cats_dict)


if __name__ == "__main__":
    main()
