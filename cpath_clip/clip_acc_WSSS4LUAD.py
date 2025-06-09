import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import tqdm
import src.open_clip as open_clip

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Label mapping: 0 for tumor, 1 for normal
label_dict = {
    "tumor": 0,
    "normal": 1
}


class PathologyDataset(Dataset):
    """
    Custom Dataset for pathology images.
    Extracts labels from filenames and applies preprocessing.
    """
    def __init__(self, img_list, preprocess):
        self.img_list = img_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")

        # Parse label from filename (assumes label is embedded as Python-style list)
        label_info = eval(Path(img_path).stem.split('-')[-1])
        label = 0 if label_info[0] == 1 else 1

        image = self.preprocess(image)
        return image, label


def load_model(model_type, pretrained_path, cache_dir):
    """
    Load OpenCLIP model and preprocessing transforms.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained_path,
        cache_dir=cache_dir
    )
    return model.cuda(), preprocess


def load_text_features(tokenizer_path, label_texts):
    """
    Encode class label descriptions into tokenized tensors.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(label_texts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=512)
    return {k: v.squeeze().cuda() for k, v in encoded.items()}


def evaluate(model, dataloader, text_features):
    """
    Run inference on the dataset and compute accuracy and classification report.
    """
    correct = 0
    total = 0
    gts, preds = [], []
    class_counts = {'tumor': 0, 'normal': 0}

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

            # Compute similarities and get predictions
            logits = (100.0 * image_feats @ text_feats.T).softmax(dim=-1)
            pred_labels = torch.argmax(logits, dim=-1)

        correct += (pred_labels == labels).sum().item()
        gts.extend(labels.cpu().numpy())
        preds.extend(pred_labels.cpu().numpy())

        for label in labels.cpu().numpy():
            class_name = 'tumor' if label == 0 else 'normal'
            class_counts[class_name] += 1
        print(correct / total)

    acc = correct / total
    print(f"Image Count by Class: {class_counts}")
    print(classification_report(gts, preds, target_names=['tumor', 'normal'], digits=3))
    print(f"Overall Accuracy: {acc:.4f}")


def main():
    # Configuration
    pretrained_path = '/mnt/Xsky/siyx/clip_prompt_eval/models/virchow/cpath_clip.pt'
    tokenizer_path = '/mnt/Xsky/syx/project/2024/model/Qwen-encoder-1.5B-tmp'
    model_type = 'ViT-L-14-336'
    cache_dir = '/mnt/Xsky/syx/model/open_clip'
    img_dir = '/mnt/Xsky/syx/dataset/pathasst_data/clip_zeroshot_cls/WSSS/1.training'
    batch_size = 128

    label_texts = [
        'An H&E patch image of tumor tissue.',
        'An H&E patch image of normal tissue.'
    ]

    # Load model and text features
    model, preprocess = load_model(model_type, pretrained_path, cache_dir)
    text_features = load_text_features(tokenizer_path, label_texts)

    # Prepare dataset and dataloader
    img_list = sorted([
        str(Path(img_dir) / f) for f in os.listdir(img_dir) if f.endswith('.png')
    ])
    random.shuffle(img_list)

    dataset = PathologyDataset(img_list, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Run evaluation
    evaluate(model, dataloader, text_features)


if __name__ == "__main__":
    main()
