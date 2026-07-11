"""Инференс YOLO-классификатора комнат на изображениях.

Пример:
    python predict.py --model model/yolo_cls_room.pt --source photos/test1.jpg
    python predict.py --model model/yolo_cls_room.pt --source photos --out outputs
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: Path):
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
    return [source]


def load_font(size: int):
    for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image_path: Path, label: str, conf: float, out_path: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font_size = max(18, img.width // 25)
    font = load_font(font_size)
    text = f"{label} {conf * 100:.1f}%"

    pad = font_size // 3
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(0, 255, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Room classification inference")
    parser.add_argument("--model", default="model/yolo_cls_room.pt", help="path to YOLO .pt model")
    parser.add_argument("--source", default="photos", help="image file or directory")
    parser.add_argument("--out", default="outputs", help="output directory")
    args = parser.parse_args()

    model = YOLO(args.model)
    images = collect_images(Path(args.source))
    out_dir = Path(args.out)

    print(f"Модель: {args.model}")
    print(f"Классы: {list(model.names.values())}\n")

    for img_path in images:
        result = model(img_path, verbose=False)[0]
        top1 = int(result.probs.top1)
        label = result.names[top1]
        conf = float(result.probs.top1conf)

        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        annotate(img_path, label, conf, out_path)
        print(f"{img_path.name:15s} -> {label:16s} {conf * 100:5.1f}%   ({out_path})")


if __name__ == "__main__":
    main()
