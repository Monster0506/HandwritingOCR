# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.6.0",
#   "torchvision>=0.21.0",
#   "transformers==4.57.1",
#   "Pillow",
#   "matplotlib",
#   "einops",
#   "addict",
#   "easydict",
#   "requests",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torchvision = { index = "pytorch-cu124" }
# ///

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract text from an image file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  uv run ocr.py scan.jpg\n  uv run ocr.py scan.jpg -o ./results/",
    )
    p.add_argument("image", help="Path to the image file")
    p.add_argument("-o", "--output", metavar="DIR",
                   help="Save result.md and result_with_boxes.jpg to this directory")
    p.add_argument("--model", default="baidu/Unlimited-OCR", metavar="MODEL",
                   help="HuggingFace model ID or local path")
    p.add_argument("--mode", choices=("gundam", "base"), default="gundam",
                   help="gundam=640px crop mode (default), base=1024px full-page mode")
    return p.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"error: file not found: {image_path}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("error: no CUDA GPU found — an NVIDIA GPU is required")

    print(f"Loading model {args.model!r} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    model = model.eval().cuda()

    image_size = 640 if args.mode == "gundam" else 1024
    crop_mode = args.mode == "gundam"

    with tempfile.TemporaryDirectory() as tmp:
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=str(image_path),
            output_path=tmp,
            base_size=1024,
            image_size=image_size,
            crop_mode=crop_mode,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )

        result_md = Path(tmp) / "result.md"
        result_img = Path(tmp) / "result_with_boxes.jpg"

        text = result_md.read_text(encoding="utf-8") if result_md.exists() else ""

        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            if result_md.exists():
                shutil.copy(result_md, out_dir / "result.md")
            if result_img.exists():
                shutil.copy(result_img, out_dir / "result_with_boxes.jpg")
            print(f"Saved to {out_dir.resolve()}", file=sys.stderr)

    print(text, end="")


if __name__ == "__main__":
    main()
