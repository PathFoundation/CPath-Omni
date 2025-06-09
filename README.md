# CPath-Omni: A Unified Multimodal Foundation Model for Patch and Whole Slide Image Analysis in Computational Pathology

This is the official repo for **CPath-Omni: A Unified Multimodal Foundation Model for Patch and Whole Slide Image Analysis in Computational Pathology** | [**Paper**](https://arxiv.org/pdf/2412.12077)

## Abstract

The emergence of large multimodal models (LMMs) has brought significant advancements to pathology. Previous research has primarily focused on separately training patch-level and whole-slide image (WSI)-level models, limiting the integration of learned knowledge across patches and WSIs and resulting in redundant models. In this work, we introduce CPath-Omni, the first 15B parameter LMM that unifies patch and WSI analysis, consolidating a variety of tasks at both levels, including classification, visual question answering, captioning, and visual referring prompting. Extensive experiments demonstrate that CPath-Omni achieves state-of-the-art (SOTA) performance across seven diverse tasks on 39 out of 42 datasets, outperforming or matching task-specific models trained for individual tasks. Additionally, we develop a specialized pathology CLIP-based visual processor for CPath-Omni, CPath-CLIP, which, for the first time, integrates different vision models and incorporates a large language model as a text encoder to build a more powerful CLIP model, which achieves SOTA performance on nine zero-shot and four few-shot datasets. Our findings highlight CPath-Omni's ability to unify diverse pathology tasks, demonstrating its potential to streamline and advance the field of foundation model in pathology.

## Usage of CPath-CLIP

### Installation

```bash
conda create -n cpath python=3.10 -y
conda activate cpath
pip install -r ./cpath_clip/requirements.txt
```



## Usage of CPath-CLIP Model

The trained CPath-CLIP can be downloaded via this [**link**](https://huggingface.co/jamessyx/CPath-CLIP).

As CPath-CLIP's vision encoder is partially initialized from Virchow2. Before using CPath-CLIP, you need to:

1. **Apply for [Virchow2 access](https://huggingface.co/paige-ai/Virchow2)** following their official licensing process
2. **Download the original Virchow2 weights**
3. **Use our provided delta weights** to reconstruct CPath-CLIP

We provide delta weights (the difference between fine-tuned CPath-CLIP Virchow2 components and original Virchow2) and [reconstruction code]() for your convenience.

```bash
python reconstruct_cpath_clip.py \
    --base models/virchow/pytorch_model.bin \
    --pretrained models/virchow/cpath_clip_minus_delta.pt \
    --delta models/virchow/delta.pt \
    --output models/virchow/cpath_clip.pt
```



The inference example is provided below, more testing examples can be find in [link]():

```python
import os
import torch
from PIL import Image
from transformers import AutoTokenizer
import src.open_clip as open_clip

# Load CPath-CLIP model
model_type = 'ViT-L-14-336'
pretrained = 'path/to/cpath_clip.pt'
model, _, preprocess = open_clip.create_model_and_transforms(
    model_type, 
    pretrained=pretrained
)
model = model.cuda()

# Define pathology labels
label_dict = {
    'lung_n': 'benign lung tissue',
    'lung_aca': 'lung adenocarcinoma', 
    'lung_scc': 'lung squamous cell carcinomas'
}

# Prepare text features using LLM-based tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/qwen-encoder-1.5B")
text_label_list = ['An image of {}'.format(label_dict[i].lower()) for i in label_dict.keys()]
texts = tokenizer(text_label_list, return_tensors='pt', padding='max_length', 
                 truncation=True, max_length=512)
text_features = {key: val.squeeze().cuda() for key, val in texts.items()}

# Single image inference
image = Image.open("lung_tissue.jpeg")
image_input = preprocess(image).unsqueeze(0).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_embeddings = model.encode_text(text_features)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute probabilities
    text_probs = (100.0 * image_features @ text_embeddings.T).softmax(dim=-1)
    
    predicted_class = torch.argmax(text_probs, dim=-1)
    confidence = torch.max(text_probs, dim=-1)[0]

print(f"Predicted class: {list(label_dict.keys())[predicted_class]}")
print(f"Confidence: {confidence:.3f}")
```



## Usage of CPath-Omni Model

The model will be released soon.



## Citation

If you use CPath-Omni in your research, please cite our paper:

```bibtex
@inproceedings{sun2025cpath,
  title={Cpath-omni: A unified multimodal foundation model for patch and whole slide image analysis in computational pathology},
  author={Sun, Yuxuan and Si, Yixuan and Zhu, Chenglu and Gong, Xuan and Zhang, Kai and Chen, Pingyi and Zhang, Ye and Shui, Zhongyi and Lin, Tao and Yang, Lin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10360--10371},
  year={2025}
}
```

## Acknowledgments

We thank the pathology community for providing valuable datasets and the open-source community for foundational tools that made this work possible.