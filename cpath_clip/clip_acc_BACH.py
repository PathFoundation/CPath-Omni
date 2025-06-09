import os
import torch
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import tqdm

import src.open_clip as open_clip

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_label_mappings():
    """
    Define label index and descriptive mappings for BACH dataset.
    """
    label_dict = {
        "A": "Benign tissue",
        "B": "In-situ carcinoma",
        "C": "Invasive carcinoma",
        "D": "Normal tissue"
    }
    return label_dict


class BACHDataset(Dataset):
    """
    BACH dataset where label is encoded in filename like: XXX_A.png
    """
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label_char = Path(path).stem.split('_')[-1][0]  # e.g., 'A'
        label = ord(label_char) - ord('A')  # A→0, B→1, C→2, D→3
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        return image, label


def prepare_text_features(label_dict, tokenizer_path):
    """
    Convert label descriptions to text embeddings.
    """
    prompts = [f"An H&E image of {desc.lower()}" for desc in label_dict.values()]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(prompts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features, label_dict):
    """
    Run zero-shot evaluation using CLIP-like similarity.
    """
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
            image_feats /= image_feats.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_feats @ text_feats.T).softmax(dim=-1)
            pred = torch.argmax(probs, dim=-1)

        correct += (pred == labels).sum().item()
        gts.extend(labels.cpu().numpy())
        preds.extend(pred.cpu().numpy())

        if total % 100 == 0:
            print(f"Running accuracy: {correct / total:.4f}")

    acc = correct / total
    print(classification_report(gts, preds, target_names=list(label_dict.values()), digits=3))
    print(f"Overall Accuracy: {acc:.4f}")


def main():
    # ===== Configs (anonymized) =====
    image_dir = Path('pathto/BACH/images')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 32

    # ===== Load model and tokenizer =====
    label_dict = get_label_mappings()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_path, cache_dir=cache_dir
    )
    model = model.cuda()
    text_features = prepare_text_features(label_dict, tokenizer_path)

    # ===== Load data =====
    image_list = sorted([str(image) for image in image_dir.iterdir() if image.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    dataset = BACHDataset(image_list, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ===== Evaluate =====
    evaluate(model, dataloader, text_features, label_dict)


if __name__ == "__main__":
    main()
