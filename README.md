# OCR

Handwriting and document OCR from the command line.

**Requirements:** NVIDIA GPU (CUDA 12.4+ drivers) · [uv](https://docs.astral.sh/uv/)

## Usage

```sh
uv run ocr.py image.jpg
```

The first run downloads the model (~8 GB) to your HuggingFace cache. Subsequent runs are fast.

Save the extracted text and an annotated image alongside the output:

```sh
uv run ocr.py image.jpg -o ./results/
```

## Options

```
positional:
  image           path to the image file

optional:
  -o, --output DIR    save result.md and result_with_boxes.jpg to DIR
  --model MODEL       HuggingFace model ID or local path
  --mode {gundam,base}
                      gundam = 640px crop mode, good for most docs (default)
                      base   = 1024px full-page mode, better for dense layouts
```

## Output

Prints extracted text to stdout. With `-o`, also writes:

- `result.md` — plain text transcript
- `result_with_boxes.jpg` — original image with detected text regions highlighted
