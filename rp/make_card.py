"""Turn a character-card JSON into a SillyTavern PNG card (and optionally install it).

    python -m rp.make_card rp/characters/su_xiaotang.json --install ~/SillyTavern/data/default-user/characters
"""
import argparse
import base64
import json
import os

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def avatar(name, size=(400, 600)):
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(40 + 120 * t), int(30 + 40 * t), int(80 + 60 * (1 - t))))
    d.ellipse([w * 0.62, h * 0.06, w * 0.9, h * 0.25], fill=(250, 236, 190))
    for i in range(9):
        x = w * (0.12 + i * 0.095)
        d.ellipse([x, h * 0.62, x + w * 0.16, h * 0.95], fill=(245, 245, 250), outline=(220, 200, 230))
    try:
        big = ImageFont.truetype(FONT, int(w * 0.42))
        small = ImageFont.truetype(FONT, int(w * 0.09))
    except OSError:
        big = small = ImageFont.load_default()
    ch = name[-1]
    bw = d.textlength(ch, font=big)
    d.text(((w - bw) / 2, h * 0.24), ch, font=big, fill=(255, 250, 240))
    sw = d.textlength(name, font=small)
    d.text(((w - sw) / 2, h * 0.53), name, font=small, fill=(255, 230, 200))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("--out", default=None)
    ap.add_argument("--install", default=None, help="SillyTavern characters dir")
    a = ap.parse_args()
    with open(a.card, encoding="utf-8") as f:
        card = json.load(f)
    name = card["data"]["name"]
    meta = PngInfo()
    meta.add_text("chara", base64.b64encode(json.dumps(card, ensure_ascii=False).encode()).decode())
    out = a.out or os.path.splitext(a.card)[0] + ".png"
    avatar(name).save(out, pnginfo=meta)
    print("wrote", out)
    if a.install:
        dst = os.path.join(os.path.expanduser(a.install), f"{name}.png")
        avatar(name).save(dst, pnginfo=meta)
        print("installed", dst)


if __name__ == "__main__":
    main()
