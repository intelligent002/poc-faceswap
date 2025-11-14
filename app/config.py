import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = BASE_DIR / "app" / "templates"
FONTS_DIR = BASE_DIR / "app" / "fonts"
MODELS_DIR = BASE_DIR / "models"

INSWAPPER_MODEL_PATH = MODELS_DIR / "inswapper_128.onnx"

# Шаблон по умолчанию
DEFAULT_TEMPLATE_ID = "template1.jpg"

# Размер итоговой картинки
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
