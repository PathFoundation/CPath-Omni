import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import tqdm

import src.open_clip as open_clip

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_label_mapping():
    """
    Define label-to-text description and label-to-index mappings.
    """
    label_text_map = {
        "NC": "Non-cancerous, Benign, Normal tissue, Non-malignant, stroma",
        "G3": "Atrophic well differentiated and dense glandular regions, Low-grade cancer, Well-differentiated glands",
        "G4": "Cribriform, ill-formed, large-fused and papillary glandular patterns, Intermediate-grade cancer, Moderately differentiated glands",
        "G5": "Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns, High-grade cancer, Poorly differentiated or undifferentiated cells"
    }

    label_index_map = {"NC": 0, "G3": 1, "G4": 2, "G5": 3}
    return label_text_map, label_index_map


def parse_excel_to_image_list(excel_path: Path, image_root: Path):
    """
    Parse Excel annotations and return image list with one-hot labels.
    """
    df = pd.read_excel(excel_path).drop(columns=['G4C'])  # Remove unused column

    label_order = ['NC', 'G3', 'G4', 'G5']
    label_groups = {label: [] for label in label_order}

    for _, row in df.iterrows():
        img_name = row['image_name']
        labels = row.values[1:]
        if np.sum(labels) == 1:
            label_idx = np.where(labels == 1)[0][0]
            label = label_order[label_idx]
            label_groups[label].append(img_name)

    image_paths = []
    img2label = {}
    for label in label_order:
        random.shuffle(label_groups[label])
        for img_name in label_groups[label]:
            img_path = str(image_root / img_name)
            image_paths.append(img_path)
            img2label[img_path] = label

    random.shuffle(image_paths)
    return image_paths, img2label


class PathologyDataset(Dataset):
    """Dataset for H&E prostate cancer classification (SICAPv2)"""
    def __init__(self, image_list, image2label, preprocess, label_index_map):
        self.image_list = image_list
        self.image2label = image2label
        self.preprocess = preprocess
        self.label_index_map = label_index_map

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.label_index_map[self.image2label[img_path]]
        return self.preprocess(image), label


def prepare_text_features(label_text_map, tokenizer_path):
    """Convert text descriptions into tokenized embeddings"""
    label_texts = [
        f"H&E histology image of {desc.lower()}"
        for desc in label_text_map.values()
    ]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(label_texts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features, label_text_map, label_index_map):
    """Run evaluation and compute classification report"""
    correct = 0
    total = 0
    gts, preds = [], []

    model.eval()
    for images, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
        images, labels = images.cuda(), labels.cuda()
        total += labels.size(0)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features_ = model.encode_text(text_features)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ text_features_.T).softmax(dim=-1)
            pred_labels = torch.argmax(logits, dim=-1)

        correct += (pred_labels == labels).sum().item()
        gts.extend(labels.cpu().numpy())
        preds.extend(pred_labels.cpu().numpy())

        if total % 20 == 0:
            print(f"Running accuracy: {correct / total:.4f}")

    acc = correct / total
    target_names = [label_text_map[label] for label in label_index_map.keys()]
    print(classification_report(gts, preds, target_names=target_names, digits=3))
    print(f"Overall Accuracy: {acc:.4f}")


def main():
    # ===== Config (anonymized paths) =====
    excel_path = Path('pathto/SICAPv2/partition/Train.xlsx')
    image_root = Path('pathto/SICAPv2/images')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 128

    # ===== Load label info and image list =====
    label_text_map, label_index_map = load_label_mapping()
    image_list, image2label = parse_excel_to_image_list(excel_path, image_root)

    # ===== Load model and tokenizer =====
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_path, cache_dir=cache_dir
    )
    model = model.cuda()
    text_features = prepare_text_features(label_text_map, tokenizer_path)

    # ===== Load data =====
    dataset = PathologyDataset(image_list, image2label, preprocess, label_index_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ===== Run evaluation =====
    evaluate(model, dataloader, text_features, label_text_map, label_index_map)


if __name__ == "__main__":
    main()
