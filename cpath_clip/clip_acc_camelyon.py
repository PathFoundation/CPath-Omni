import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import tqdm

import src.open_clip as open_clip

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_label_mapping():
    """
    Return natural language prompt → class index mapping.
    """
    return {
        'an H&E patch of lymph nodes containing metastases': 0,
        'an H&E patch of lymph nodes without any metastases': 1
    }


class PathologyDataset(Dataset):
    """
    Dataset containing image paths and their corresponding captions.
    """
    def __init__(self, image_list, caption_list, preprocess, label_dict):
        self.image_list = image_list
        self.caption_list = caption_list
        self.preprocess = preprocess
        self.label_dict = label_dict

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        image = self.preprocess(image)
        label = self.label_dict[self.caption_list[idx]]
        return image, label


def prepare_text_features(text_prompts, tokenizer_path):
    """
    Tokenize a list of text prompts using specified tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(text_prompts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features):
    """
    Run evaluation loop and return accuracy.
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

            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_feats @ text_feats.T).softmax(dim=-1)
            pred_labels = torch.argmax(logits, dim=-1)

        correct += (pred_labels == labels).sum().item()
        gts.extend(labels.cpu().numpy())
        preds.extend(pred_labels.cpu().numpy())

        if total % 100 == 0:
            print(f"Running accuracy: {correct / total:.4f}")

    acc = correct / total
    print(f"Final Accuracy: {acc:.4f}")
    return acc


def main():
    # ===== Paths and Configs (anonymized) =====
    data_path = Path('pathto/camelyon/dict_for_camelyon17_new.npy')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 256

    # ===== Text prompts =====
    text_prompts = [
        'this is an H&E image of lymph node metastasis presented in image',
        'this is an H&E image of normal lymph node presented in image'
    ]

    label_dict = get_label_mapping()

    # ===== Load model =====
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_path, cache_dir=cache_dir
    )
    model = model.cuda()

    # ===== Load data =====
    data = np.load(data_path, allow_pickle=True).item()
    image_list, caption_list = data['img'], data['caption']
    pairs = list(zip(image_list, caption_list))
    random.shuffle(pairs)
    image_list, caption_list = zip(*pairs)

    # ===== Dataset & Loader =====
    dataset = PathologyDataset(image_list, caption_list, preprocess, label_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ===== Text encoding =====
    text_features = prepare_text_features(text_prompts, tokenizer_path)

    # ===== Evaluate =====
    evaluate(model, dataloader, text_features)


if __name__ == "__main__":
    main()
