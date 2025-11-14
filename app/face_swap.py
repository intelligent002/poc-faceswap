from pathlib import Path
import cv2
import numpy as np
import insightface

from .config import INSWAPPER_MODEL_PATH
from .face_enhance import FaceEnhancer


class FaceSwapEngine:
    """
    CPU-only face swap + post-enhancement using GFPGAN-ONNX
    """

    def __init__(self):
        print("[FaceSwap] Initializing FaceAnalysis...")

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[FaceSwap] FaceAnalysis loaded")

        # Check swapper model
        if not INSWAPPER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"InSwapper model missing: {INSWAPPER_MODEL_PATH}"
            )

        print(f"[FaceSwap] Loading InSwapper: {INSWAPPER_MODEL_PATH}")

        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_MODEL_PATH),
            download=False,
            download_zip=False,
            providers=["CPUExecutionProvider"],
        )
        print("[FaceSwap] InSwapper loaded successfully")

        # --- NEW: Load GFPGAN enhancer ---
        enhancer_path = Path("models/GFPGANv1.4.onnx")
        if enhancer_path.exists():
            self.enhancer = FaceEnhancer(str(enhancer_path))
            print("[FaceSwap] FaceEnhancer enabled")
        else:
            self.enhancer = None
            print("[FaceSwap] No gfpgan.onnx — enhancer disabled")

    def _decode(self, data: bytes):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img

    def _main_face(self, img):
        faces = self.face_app.get(img)
        if not faces:
            raise ValueError("No face detected")
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

    def swap_face(self, user_photo_bytes, template_bytes):
        user_img = self._decode(user_photo_bytes)
        template_img = self._decode(template_bytes)

        src_face = self._main_face(user_img)
        dst_face = self._main_face(template_img)

        # InSwapper (face swap)
        out = self.swapper.get(
            template_img,
            dst_face,
            src_face,
            paste_back=True
        )

        if out is None:
            raise RuntimeError("Face swap failed")

        # --- NEW: Face enhancement ---
        if self.enhancer:
            try:
                x1, y1, x2, y2 = dst_face.bbox.astype(int)
                face_crop = out[y1:y2, x1:x2]

                enhanced = self.enhancer.enhance_face(face_crop)

                # Replace the enhanced face back
                out[y1:y2, x1:x2] = enhanced
            except Exception as e:
                print("[FaceEnhancer] error:", e)

        return out
