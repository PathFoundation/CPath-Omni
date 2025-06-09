import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import tqdm

import src.open_clip as open_clip

# Set visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_label_mappings():
    """Return class-to-description and class-to-index mappings"""
    label_dict = {
        "nontumor_skin_chondraltissue_chondraltissue": "Non-tumor chondral tissue",
        "nontumor_skin_dermis_dermis": "Non-tumor dermis",
        "nontumor_skin_elastosis_elastosis": "Non-tumor elastosis",
        "nontumor_skin_epidermis_epidermis": "Non-tumor epidermis",
        "nontumor_skin_hairfollicle_hairfollicle": "Non-tumor hair follicle",
        "nontumor_skin_muscle_skeletal": "Non-tumor skeletal muscle",
        "nontumor_skin_necrosis_necrosis": "Non-tumor necrosis",
        "nontumor_skin_nerves_nerves": "Non-tumor nerves",
        "nontumor_skin_sebaceousglands_sebaceousglands": "Non-tumor sebaceous glands",
        "nontumor_skin_subcutis_subcutis": "Non-tumor subcutis",
        "nontumor_skin_sweatglands_sweatglands": "Non-tumor sweat glands",
        "nontumor_skin_vessel_vessel": "Non-tumor vessel",
        "tumor_skin_epithelial_bcc": "Tumor epithelial basal cell carcinoma",
        "tumor_skin_epithelial_sqcc": "Tumor epithelial squamous cell carcinoma",
        "tumor_skin_melanoma_melanoma": "Tumor melanoma",
        "tumor_skin_naevus_naevus": "Tumor naevus"
    }

    label_index_dict = {k: i for i, k in enumerate(label_dict.keys())}
    return label_dict, label_index_dict


class SkinCancerDataset(Dataset):
    """Skin pathology image dataset based on file list and label mapping"""
    def __init__(self, img_list, img2label, preprocess, label_index_dict):
        self.img_list = img_list
        self.img2label = img2label
        self.preprocess = preprocess
        self.label_index_dict = label_index_dict

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")
        label_name = self.img2label[img_path]
        label = self.label_index_dict[label_name]
        return self.preprocess(image), label


def load_model(model_type, pretrained_path, cache_dir):
    """Load OpenCLIP model and preprocessing transforms"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained_path,
        cache_dir=cache_dir
    )
    return model.cuda(), preprocess


def prepare_text_features(label_dict, tokenizer_path):
    """Convert label descriptions to tokenized text embeddings"""
    label_texts = [f"An H&E image of {desc.lower()}" for desc in label_dict.values()]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(label_texts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features, label_dict, label_index_dict):
    """Run inference and classification evaluation"""
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

            # Normalize features
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            # Cosine similarity and classification
            probs = (100.0 * image_feats @ text_feats.T).softmax(dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)

        correct += (pred_labels == labels).sum().item()
        gts.extend(labels.cpu().numpy())
        preds.extend(pred_labels.cpu().numpy())

    acc = correct / total
    target_names = [label_dict[k] for k in label_index_dict.keys()]
    print(classification_report(gts, preds, target_names=target_names, digits=3))
    print(f"Overall Accuracy: {acc:.4f}")


def main():
    # ====== Configurable paths (anonymized) ======
    csv_path = Path('pathto/skincancer/raw/tiles-v2.csv')
    root_dir = Path('pathto/skincancer/raw')
    pretrained_path = 'pathto/cpath_clip.pt'
    tokenizer_path = './Qwen-encoder-1.5B'
    model_type = 'ViT-L-14-336'
    cache_dir = 'pathto/open_clip'
    batch_size = 128

    # ====== Load labels and data ======
    label_dict, label_index_dict = load_label_mappings()

    df = pd.read_csv(csv_path)
    img_list = []
    img2label = {}
    for _, row in df.iterrows():
        img_rel = row['file'].replace('data/', '')
        label = row['class']
        img_path = str(root_dir / img_rel)
        img_list.append(img_path)
        img2label[img_path] = label
    random.shuffle(img_list)

    # ====== Load model and data ======
    model, preprocess = load_model(model_type, pretrained_path, cache_dir)
    text_features = prepare_text_features(label_dict, tokenizer_path)

    dataset = SkinCancerDataset(img_list, img2label, preprocess, label_index_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ====== Evaluate model ======
    evaluate(model, dataloader, text_features, label_dict, label_index_dict)


if __name__ == "__main__":
    main()
