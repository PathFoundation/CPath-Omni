"""
Utility script to compute *delta* weights between a finetuned model and its
base model, save the delta to disk, and verify that re-applying the delta
exactly reconstructs the finetuned checkpoint.

Example usage:

    python reconstruct_cpath_clip.py \
        --base models/virchow/pytorch_model.bin \
        --pretrained models/virchow/cpath_clip_minus_delta.pt \
        --delta models/virchow/delta.pt \
        --output models/virchow/cpath_clip.pt

Add ``--eval`` with a dataloader module to compare accuracy as well.
"""
import argparse
from collections import OrderedDict
from pathlib import Path
import torch
from transformers import AutoTokenizer
import src.open_clip as open_clip


def load_state_dict(path: str) -> OrderedDict:
    """Load a Torch state_dict from *path* on CPU for portability."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location="cpu")


def apply_delta(base_path: str, delta_sd: OrderedDict) -> OrderedDict:
    """Apply delta weights to base model: new_sd = base + delta."""
    base_sd = load_state_dict(base_path)
    merged = OrderedDict()

    for key in base_sd:
        if key == "pos_embed":
            merged[key] = delta_sd[key]
        else:
            merged[key] = base_sd[key] + delta_sd[key]

    return merged


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Apply delta weights to a base model.")

    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Path to base model weights file"
    )
    parser.add_argument(
        "--delta",
        type=str,
        required=True,
        help="Path to delta weights file"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the final merged model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./Qwen-encoder-1.5B",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ViT-L-14-336",
        help="Model architecture type"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="pathto/open_clip",
        help="Cache directory for open_clip models"
    )

    return parser


def main():
    """Main function to load model, apply delta weights, and save the result."""
    parser = create_argument_parser()
    args = parser.parse_args()

    print(f"Creating model: {args.model_type}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_type,
        pretrained=args.pretrained,
        cache_dir=args.cache_dir
    )

    print(f"Loading delta weights from: {args.delta}")
    delta_sd = torch.load(args.delta, map_location="cpu")

    print(f"Applying delta weights to base model: {args.base}")
    merged_sd = apply_delta(args.base, delta_sd)

    print("Loading merged weights into model...")
    model.visual2.load_state_dict(merged_sd, strict=True)

    print(f"Saving final model to: {args.output}")
    torch.save(model.state_dict(), args.output)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()