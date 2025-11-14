from pathlib import Path
import os

import cv2
import numpy as np
import insightface
import onnxruntime as ort

# Попробуем использовать skimage для histogram matching
try:
    from skimage.exposure import match_histograms
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

# ====================== Глобальная оптимизация OpenCV ======================

cv2.setUseOptimized(True)
try:
    # ограничимся 4 потоками, чтобы не убивать систему
    cv2.setNumThreads(min(4, os.cpu_count() or 4))
except Exception:
    pass


# =====================================================================
# ------------------------- ONNX UTILS --------------------------------
# =====================================================================

def get_onnx_providers():
    """
    Выбираем максимально быстрые доступные провайдеры.
    Если есть OpenVINOExecutionProvider — используем его + CPU.
    Иначе — только CPUExecutionProvider.
    """
    available = ort.get_available_providers()
    providers = []

    if "OpenVINOExecutionProvider" in available:
        providers.append("OpenVINOExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        providers = ["CPUExecutionProvider"]

    return providers


def create_onnx_session(onnx_path: str) -> ort.InferenceSession:
    """
    Создаёт InferenceSession с включённой максимальной оптимизацией графа.
    """
    sess_options = ort.SessionOptions()
    level = getattr(
        ort.GraphOptimizationLevel,
        "ORT_ENABLE_ALL",
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_options.graph_optimization_level = level

    return ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=get_onnx_providers()
    )


# =====================================================================
# ------------------------- UTILITY FUNCTIONS -------------------------
# =====================================================================

def safe_letterbox_to_512(img):
    """
    Resize с сохранением пропорций в квадрат 512×512.
    Гарантирует выход (512,512,3), даже если на входе мусор.
    """
    if img is None or img.size == 0:
        print("[Letterbox] invalid input, returning blank 512")
        return np.zeros((512, 512, 3), dtype=np.uint8), (0, 0, 512, 512)

    h, w = img.shape[:2]

    # Слишком маленькие лица – пропускаем
    if h < 20 or w < 20:
        print(f"[Letterbox] face too small {img.shape}, returning blank 512")
        return np.zeros((512, 512, 3), dtype=np.uint8), (0, 0, 512, 512)

    size = 512
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    nh = max(min(nh, 512), 1)
    nw = max(min(nw, 512), 1)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    return canvas, (top, left, nh, nw)


def color_match_hist(dst_face, src_face):
    """
    Приводим enhanced-лицо (src_face) к цветам оригинального лица (dst_face).
    Если skimage нет — fallback на LAB-transfer.
    """
    if HAVE_SKIMAGE:
        matched = match_histograms(src_face, dst_face, channel_axis=-1)
        return matched.astype(np.uint8)

    # Fallback: LAB transfer
    src_lab = cv2.cvtColor(src_face, cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(dst_face, cv2.COLOR_BGR2LAB)

    src_mean, src_std = cv2.meanStdDev(src_lab)
    dst_mean, dst_std = cv2.meanStdDev(dst_lab)

    dst_norm = ((dst_lab - dst_mean) * (src_std / (dst_std + 1e-6))) + src_mean
    dst_norm = np.clip(dst_norm, 0, 255).astype(np.uint8)

    return cv2.cvtColor(dst_norm, cv2.COLOR_LAB2BGR)


def make_ellipse_mask(h, w, blur_ksize=51, blur_sigma=20, erode_px=10):
    """
    Эллиптическая маска + эрозия + blur для мягких границ.
    Уменьшены размеры ядра и sigma, чтобы не грузить CPU.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (w // 2, h // 2),
        (w // 2, h // 2),
        0, 0, 360,
        255, -1
    )

    if erode_px > 0:
        k = max(1, erode_px)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    # blur_ksize должен быть нечётный
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), blur_sigma)
    return mask


def adaptive_sharpen(face_bgr):
    """
    Лёгкий unsharp-mask + усиление только в зонах с границами (Canny).
    """
    # базовый unsharp
    blur = cv2.GaussianBlur(face_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(face_bgr, 1.5, blur, -0.5, 0)

    # карта граней
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    # нормализуем в [0,1]
    mask = edges.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = np.clip(mask, 0.0, 1.0)
    mask = mask[..., None]         # H×W×1

    # усиливаем детали только там, где mask > 0
    out = sharpened.astype(np.float32) * mask + face_bgr.astype(np.float32) * (1 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


# =====================================================================
# --------------------------- RealESRGAN ONNX --------------------------
# =====================================================================

class RealESRGAN_ONNX:
    def __init__(self, onnx_path: str):
        self.sess = create_onnx_session(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name

    def upscale(self, img):
        if img is None or img.size == 0:
            return img

        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]
        out = self.sess.run(None, {self.input_name: x})[0]

        out = np.squeeze(out)
        out = np.transpose(out, (1, 2, 0))
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# =====================================================================
# -------------------------- FACE SWAP ENGINE -------------------------
# =====================================================================

class FaceSwapEngine:

    def __init__(self):
        print("[FaceSwap] Initializing FaceAnalysis...")

        # Используем более мелкий размер детекции — сильный буст по скорости
        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=get_onnx_providers(),
        )
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))
        print("[FaceSwap] FaceAnalysis loaded")

        # ---------------------- InSwapper -----------------------
        model_path = Path("models/inswapper_128.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"InSwapper model not found: {model_path}")

        print(f"[FaceSwap] Loading InSwapper: {model_path}")
        self.swapper = insightface.model_zoo.get_model(
            str(model_path),
            providers=get_onnx_providers(),
            download=False,
            download_zip=False,
        )
        print("[FaceSwap] InSwapper loaded")

        # ---------------------- GFPGAN Enhancer -----------------
        from app.face_enhance import FaceEnhancer   # важно: пакет app
        gfp = Path("models/GFPGANv1.4.onnx")
        self.gfpgan = FaceEnhancer(str(gfp)) if gfp.exists() else None
        print("[FaceSwap] GFPGAN =", "enabled" if self.gfpgan else "disabled")

        # ---------------------- RealESRGAN ----------------------
        esr_path = Path("models/RealESRGAN_x4plus/model.onnx")
        self.esr = RealESRGAN_ONNX(str(esr_path)) if esr_path.exists() else None
        print("[FaceSwap] RealESRGAN =", "enabled" if self.esr else "disabled")

    # =================================================================

    def _decode(self, data: bytes):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img

    def _main_face(self, img):
        faces = self.face_app.get(img)
        if not faces:
            raise ValueError("No face detected")
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # =================================================================

    def _enhance_face_patch(self, face_patch):
        """
        Внутренний шаг: GFPGAN + (опционально ESRGAN) + color match + sharpen.
        На входе: кроп из уже свапнутого изображения.

        Оптимизация:
        - Для крупных лиц (>= 220px) пропускаем тяжёлый GFPGAN/ESRGAN
          и делаем только быстрый adaptive_sharpen.
        """
        h, w = face_patch.shape[:2]
        if h < 40 or w < 40:
            print("[Enhance] face too small, skipping")
            return face_patch

        # Быстрый путь для больших лиц — только шарпинг
        if max(h, w) >= 220:
            return adaptive_sharpen(face_patch)

        # 1) letterbox → 512
        face_512, meta = safe_letterbox_to_512(face_patch)

        # 2) GFPGAN (мягко, с fallback внутри FaceEnhancer)
        if self.gfpgan:
            gfp_512 = self.gfpgan.enhance_face(face_512)
        else:
            gfp_512 = face_512

        # 3) ESRGAN ×4 (если подключён) — можно при желании отключить
        if self.esr:
            upscaled = self.esr.upscale(gfp_512)
        else:
            upscaled = gfp_512

        # 4) “unletterbox”
        top, left, nh, nw = meta
        unboxed = upscaled[top:top + nh, left:left + nw]

        # 5) resize обратно в размер исходного bbox
        enhanced_resized = cv2.resize(
            unboxed,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        # 6) цвет под оригинальное лицо
        matched = color_match_hist(face_patch, enhanced_resized)

        # 7) адаптивный шарпинг деталей
        sharpened = adaptive_sharpen(matched)

        # 8) мягкое смешивание с оригиналом (чтобы не было “переполировки”)
        out = (
            face_patch.astype(np.float32) * 0.70 +  # more original, less artifacts
            sharpened.astype(np.float32) * 0.30
        )
        return np.clip(out, 0, 255).astype(np.uint8)

    # =================================================================

    def swap_face(self, user_photo_bytes, template_bytes):
        """
        Основной публичный метод: байты user → байты template → готовое изображение BGR.
        """
        user_img = self._decode(user_photo_bytes)
        template_img = self._decode(template_bytes)

        src_face = self._main_face(user_img)
        dst_face = self._main_face(template_img)

        # -------------------- FACE SWAP ------------------------------
        # paste_back=True → получаем уже геометрически корректный swap,
        # дальше только улучшаем и аккуратно встраиваем.
        swapped_full = self.swapper.get(
            template_img,
            dst_face,
            src_face,
            paste_back=True,
        )

        if swapped_full is None:
            raise RuntimeError("Face swap failed")

        # Если нет улучшайзеров — просто возвращаем
        if not self.gfpgan and not self.esr:
            return swapped_full

        # ----------------- ENHANCEMENT + FAST BLEND ---------------
        try:
            x1, y1, x2, y2 = dst_face.bbox.astype(int)

            # safety clamp
            h_full, w_full = swapped_full.shape[:2]
            x1 = max(0, min(x1, w_full - 1))
            x2 = max(1, min(x2, w_full))
            y1 = max(0, min(y1, h_full - 1))
            y2 = max(1, min(y2, h_full))

            face_patch = swapped_full[y1:y2, x1:x2]

            enhanced_patch = self._enhance_face_patch(face_patch)

            ph, pw = enhanced_patch.shape[:2]

            # Маска для alpha-blend (одноканальная, 8-бит, 0–255)
            mask = make_ellipse_mask(ph, pw)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]  # H×W×1

            # быстрый альфа-блендинг вместо медленного seamlessClone
            roi = swapped_full[y1:y2, x1:x2].astype(np.float32)
            enh = enhanced_patch.astype(np.float32)

            blended = enh * alpha + roi * (1.0 - alpha)
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            swapped_full[y1:y2, x1:x2] = blended

            return swapped_full

        except Exception as e:
            print("[FaceSwap] enhancement/blending error:", e)
            # хоть что-то вернуть
            return swapped_full
