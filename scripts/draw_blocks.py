"""D1 check: draw MinerU block bboxes on a rendered PDF page.

Usage: python scripts/draw_blocks.py <middle.json> <file.pdf> <pdf_page (1-based)> <out.png>
"""
import json
import sys

import pypdfium2 as pdfium
from PIL import ImageDraw

COLORS = {
    "table": (220, 30, 30),
    "image": (30, 120, 220),
    "paragraph_title": (30, 160, 60),
    "text": (150, 150, 150),
}
DPI = 150


def main(middle_path, pdf_path, pdf_page, out_path):
    middle = json.load(open(middle_path))
    page_idx = pdf_page - 1  # middle.json page_idx is 0-based
    page = next(p for p in middle["pages"] if p["page_idx"] == page_idx)
    size = next(
        s for s in middle["extensions"]["docvortex_layout"]["pages"] if s["page_idx"] == page_idx
    )

    img = pdfium.PdfDocument(pdf_path)[page_idx].render(scale=DPI / 72).to_pil().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for block in page["blocks"]:
        # bboxes are normalised to 0-1 of the page; points = value * width_pt / height_pt
        x0, y0, x1, y1 = block["bbox"]
        width = 5 if block["type"] == "table" else 2
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=COLORS.get(block["type"], (200, 150, 0)), width=width)
        if block["type"] == "table":
            pts = [x0 * size["width_pt"], y0 * size["height_pt"], x1 * size["width_pt"], y1 * size["height_pt"]]
            print(f"table index={block['index']} bbox_pt={[round(v, 1) for v in pts]}")
    img.save(out_path)
    print(f"saved {out_path} ({w}x{h})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
