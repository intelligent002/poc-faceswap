from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from .face_swap import FaceSwapEngine
from .overlay import apply_roomvu_overlay
from .config import TEMPLATES_DIR, DEFAULT_TEMPLATE_ID

app = FastAPI(title="CPU Face Swap Service")

engine = FaceSwapEngine()


def load_template_bytes(template_id: Optional[str]) -> bytes:
    name = template_id or DEFAULT_TEMPLATE_ID
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")
    return path.read_bytes()


@app.post("/generate", summary="Generate Roomvu-style banner")
async def generate_banner(
    user_photo: UploadFile = File(...),
    template_id: Optional[str] = Form(None),
    headline: str = Form("Vancouver rent drops 8%"),
    source: str = Form("Link Newspaper"),
    name: str = Form("Arkady Rosenberg"),
    phone: str = Form("+1 778 239 4992"),
):
    try:
        user_bytes = await user_photo.read()
        template_bytes = load_template_bytes(template_id)

        swapped_bgr = engine.swap_face(user_bytes, template_bytes)

        final_jpeg = apply_roomvu_overlay(
            swapped_bgr,
            headline=headline,
            source=source,
            name=name,
            phone=phone,
        )

        return Response(content=final_jpeg, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
