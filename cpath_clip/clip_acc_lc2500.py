import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import tqdm

import src.open_clip as open_clip

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_label_mappings():
    """
    Return label name → description and label name → index mappings.
    """
    label_dict = {
        'lung_n': 'benign lung tissue',
        'lung_aca': 'lung adenocarcinoma',
        'lung_scc': 'lung squamous cell carcinomas'
    }

    label_index_dict = {
        'lung_n': 0,
        'lung_aca': 1,
        'lung_scc': 2
    }

    return label_dict, label_index_dict


def walk_dir(root: Path, exts=['.jpeg']):
    """
    Recursively collect all image paths under root directory.
    """
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                paths.append(str(Path(dirpath) / f))
    return paths


class PathologyDataset(Dataset):
    """
    Pathology dataset where folder name indicates the label.
    """
    def __init__(self, image_paths, label_index_dict, preprocess):
        self.image_paths = image_paths
        self.label_index_dict = label_index_dict
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        folder_name = Path(img_path).parent.name
        label = self.label_index_dict[folder_name]
        image = Image.open(img_path).convert("RGB")
        return self.preprocess(image), label


def prepare_text_features(label_dict, tokenizer_path):
    """
    Encode text prompts based on label descriptions.
    """
    text_prompts = [f"An image of {desc.lower()}" for desc in label_dict.values()]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(text_prompts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features, label_dict):
    """
    Perform zero-shot evaluation.
    """
    correct = 0
    total = 0
    gts, preds = [], []

    model.eval()
    for images, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
        images, labels = images.cuda(), labels.cuda()
        total += labels.size(0)

        with torch.no_grad():
            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(text_features)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
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
    # ====== Configs (anonymized) ======
    image_root = Path('pathto/lung_image_sets')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 128

    # ====== Load labels and image list ======
    label_dict, label_index_dict = get_label_mappings()
    image_list = walk_dir(image_root, exts=['.jpeg'])
    random.shuffle(image_list)

    # ====== Load model and tokenizer ======
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_path, cache_dir=cache_dir
    )
    model = model.cuda()
    text_features = prepare_text_features(label_dict, tokenizer_path)

    # ====== Build dataset and evaluate ======
    dataset = PathologyDataset(image_list, label_index_dict, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    evaluate(model, dataloader, text_features, label_dict)


if __name__ == "__main__":
    main()
