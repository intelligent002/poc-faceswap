from io import BytesIO
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from .config import FONTS_DIR, OUTPUT_WIDTH  # OUTPUT_WIDTH = max ширина, например 1024


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS_DIR / name
    if not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    # Pillow 10+: textbbox вместо textsize
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def apply_roomvu_overlay(
    bgr_image,
    headline: str,
    source: str,
    name: str,
    phone: str,
) -> bytes:
    """
    Принимает BGR (OpenCV), возвращает JPEG bytes.
    - Сохраняет исходный aspect ratio
    - Масштабирует только по ширине (до OUTPUT_WIDTH, если нужно)
    - Рисует плашку ~25% высоты снизу
    """

    import cv2

    # --- BGR -> PIL Image ---
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGB")

    orig_w, orig_h = img.size

    # --- Масштабируем по ширине, сохраняем пропорции ---
    target_w = min(OUTPUT_WIDTH, orig_w)  # не увеличиваем мелкие изображения
    scale = target_w / orig_w
    target_h = int(orig_h * scale)

    if (target_w, target_h) != (orig_w, orig_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    W, H = img.size

    # --- Плашка ---
    overlay_height = int(H * 0.25)
    overlay_y0 = H - overlay_height

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    overlay_color = (25, 70, 120, 230)  # тёмно-синий
    odraw.rectangle([(0, overlay_y0), (W, H)], fill=overlay_color)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # --- Шрифты ---
    font_headline = _load_font("OpenSans-Bold.ttf", 42)
    font_source = _load_font("OpenSans-Regular.ttf", 24)
    font_name = _load_font("OpenSans-Bold.ttf", 30)
    font_phone = _load_font("OpenSans-Regular.ttf", 24)

    white = (255, 255, 255)
    padding_x = 40
    padding_y = overlay_y0 + 25

    # --- Текст: верх плашки ---
    draw.text(
        (padding_x, padding_y),
        headline,
        font=font_headline,
        fill=white,
    )

    draw.text(
        (padding_x, padding_y + 55),
        source,
        font=font_source,
        fill=white,
    )

    # --- Нижняя строка: имя слева, телефон справа ---
    name_y = H - 60
    draw.text(
        (padding_x, name_y),
        name,
        font=font_name,
        fill=white,
    )

    phone_text = phone
    w_phone, _ = _text_size(draw, phone_text, font_phone)
    draw.text(
        (W - w_phone - padding_x, name_y + 5),
        phone_text,
        font=font_phone,
        fill=white,
    )

    # --- JPEG выход ---
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
