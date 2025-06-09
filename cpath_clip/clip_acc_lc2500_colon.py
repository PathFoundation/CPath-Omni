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
    Define text descriptions and index mapping for colon dataset.
    """
    label_dict = {
        'colon_n': 'normal colon tissue',
        'colon_aca': 'colon adenocarcinoma'
    }

    label_index_dict = {
        'colon_n': 0,
        'colon_aca': 1
    }

    return label_dict, label_index_dict


def walk_dir(root: Path, exts=['.jpeg']):
    """
    Recursively collect all image paths under root with given extensions.
    """
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                paths.append(str(Path(dirpath) / f))
    return paths


class PathologyDataset(Dataset):
    """
    Dataset where each image is labeled based on its parent folder name.
    Folder names must match label_index_dict keys.
    """
    def __init__(self, image_paths, label_index_dict, preprocess):
        self.image_paths = image_paths
        self.label_index_dict = label_index_dict
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label_folder = Path(path).parent.name
        label = self.label_index_dict[label_folder]
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        return image, label


def prepare_text_features(label_dict, tokenizer_path):
    """
    Tokenize textual descriptions of labels.
    """
    texts = [f"An image of {desc.lower()}" for desc in label_dict.values()]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(texts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features, label_dict):
    """
    Run zero-shot classification and print metrics.
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
    image_root = Path('pathto/colon_image_sets')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 128

    # ====== Labels and data ======
    label_dict, label_index_dict = get_label_mappings()
    image_list = walk_dir(image_root, ['.jpeg'])
    random.shuffle(image_list)

    # ====== Model and tokenizer ======
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_path, cache_dir=cache_dir
    )
    model = model.cuda()
    text_features = prepare_text_features(label_dict, tokenizer_path)

    # ====== Dataset and loader ======
    dataset = PathologyDataset(image_list, label_index_dict, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ====== Evaluation ======
    evaluate(model, dataloader, text_features, label_dict)


if __name__ == "__main__":
    main()
