"""
One-off helper: generates a placeholder app icon and splash screen
for the Kerala IT Hub Android app using Pillow.

Run once with:  python android_app/generate_assets.py

Feel free to replace android_app/assets/icon.png and
android_app/assets/presplash.png with real artwork later --
buildozer.spec just needs files at those paths.
"""

import os

from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

BRAND_BG = (13, 71, 161)       # deep blue
BRAND_ACCENT = (255, 193, 7)   # amber


def load_font(size):

    candidates = [
        "arialbd.ttf",
        "DejaVuSans-Bold.ttf",
    ]

    for name in candidates:

        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    return ImageFont.load_default()


def make_icon():

    size = 512

    img = Image.new("RGBA", (size, size), BRAND_BG)
    draw = ImageDraw.Draw(img)

    margin = 28
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=90,
        outline=BRAND_ACCENT,
        width=14
    )

    font = load_font(190)
    text = "KH"

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    draw.text(
        ((size - text_w) / 2 - bbox[0], (size - text_h) / 2 - bbox[1]),
        text,
        font=font,
        fill=(255, 255, 255)
    )

    img.save(os.path.join(ASSETS_DIR, "icon.png"))


def make_presplash():

    w, h = 720, 1280

    img = Image.new("RGBA", (w, h), BRAND_BG)
    draw = ImageDraw.Draw(img)

    title_font = load_font(72)
    subtitle_font = load_font(32)

    title = "KERALA IT HUB"
    subtitle = "AI Course & Institute Navigator"

    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = bbox[2] - bbox[0]

    draw.text(
        ((w - title_w) / 2, h / 2 - 80),
        title,
        font=title_font,
        fill=(255, 255, 255)
    )

    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_w = bbox[2] - bbox[0]

    draw.text(
        ((w - subtitle_w) / 2, h / 2 + 20),
        subtitle,
        font=subtitle_font,
        fill=BRAND_ACCENT
    )

    img.save(os.path.join(ASSETS_DIR, "presplash.png"))


if __name__ == "__main__":

    os.makedirs(ASSETS_DIR, exist_ok=True)

    make_icon()
    make_presplash()

    print(f"Icon and presplash written to {ASSETS_DIR}")
