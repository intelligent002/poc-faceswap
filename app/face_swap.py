from pathlib import Path
import os
import time

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

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), blur_sigma)
    return mask


def adaptive_sharpen(face_bgr):
    """
    Лёгкий unsharp-mask + усиление только в зонах с границами (Canny).
    Быстро и довольно красиво.
    """
    blur = cv2.GaussianBlur(face_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(face_bgr, 1.5, blur, -0.5, 0)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    mask = edges.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = np.clip(mask, 0.0, 1.0)
    mask = mask[..., None]

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
        t0 = time.perf_counter()

        # -------- режим работы: fast / quality --------
        self.mode = os.getenv("FACESWAP_MODE", "fast").lower()
        if self.mode not in ("fast", "quality"):
            self.mode = "fast"
        print(f"[FaceSwap] Mode: {self.mode}")

        # ---------------- FaceAnalysis -----------------
        print("[FaceSwap] Initializing FaceAnalysis...")
        t = time.perf_counter()
        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",                  # оставляем Buffalo_l, но режем det_size
            providers=get_onnx_providers(),
        )
        # меньше размер детекции → заметно быстрее
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))
        t_det = (time.perf_counter() - t) * 1000
        print(f"[FaceSwap] FaceAnalysis loaded in {t_det:.1f} ms")

        # ---------------------- InSwapper -----------------------
        model_path = Path("models/inswapper_128.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"InSwapper model not found: {model_path}")

        print(f"[FaceSwap] Loading InSwapper: {model_path}")
        t = time.perf_counter()
        self.swapper = insightface.model_zoo.get_model(
            str(model_path),
            providers=get_onnx_providers(),
            download=False,
            download_zip=False,
        )
        t_sw = (time.perf_counter() - t) * 1000
        print(f"[FaceSwap] InSwapper loaded in {t_sw:.1f} ms")

        # ---------------------- Enhancer (GFPGAN) -----------------
        # В fast-режиме НЕ грузим тяжёлый GAN вообще
        self.gfpgan = None
        if self.mode == "quality":
            from app.face_enhance import FaceEnhancer
            gfp = Path("models/GFPGANv1.4.onnx")
            if gfp.exists():
                print(f"[FaceSwap] Loading GFPGAN: {gfp}")
                t = time.perf_counter()
                self.gfpgan = FaceEnhancer(str(gfp))
                t_g = (time.perf_counter() - t) * 1000
                print(f"[FaceSwap] GFPGAN loaded in {t_g:.1f} ms")
            else:
                print("[FaceSwap] GFPGAN model not found, enhancer disabled")

        # ---------------------- RealESRGAN ----------------------
        # По умолчанию выключаем, можно включить через ENV
        self.esr = None
        enable_esr = os.getenv("ENABLE_ESRGAN", "0") == "1"
        if self.mode == "quality" and enable_esr:
            esr_path = Path("models/RealESRGAN_x4plus/model.onnx")
            if esr_path.exists():
                print(f"[FaceSwap] Loading RealESRGAN: {esr_path}")
                t = time.perf_counter()
                self.esr = RealESRGAN_ONNX(str(esr_path))
                t_e = (time.perf_counter() - t) * 1000
                print(f"[FaceSwap] RealESRGAN loaded in {t_e:.1f} ms")
            else:
                print("[FaceSwap] RealESRGAN model not found, ESR disabled")

        t_total = (time.perf_counter() - t0) * 1000
        print(f"[FaceSwap] Engine initialized in {t_total:.1f} ms")

    # =================================================================

    def _decode(self, data: bytes, label: str):
        t0 = time.perf_counter()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image bytes: {label}")
        dt = (time.perf_counter() - t0) * 1000
        print(f"[FS] DECODE {label}: {dt:.1f} ms")
        return img

    def _main_face(self, img, label: str):
        t0 = time.perf_counter()
        faces = self.face_app.get(img)
        if not faces:
            raise ValueError(f"No face detected in {label}")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        dt = (time.perf_counter() - t0) * 1000
        print(f"[FS] DETECT {label}: {dt:.1f} ms")
        return face

    # =================================================================

    def _enhance_face_patch(self, face_patch):
        """
        Внутренний шаг: улучшение лица.
        В fast-режиме: только adaptive_sharpen (без GFPGAN/ESRGAN).
        В quality-режиме: полный пайплайн GFPGAN(+ESR) + color match + sharpen.
        """
        h, w = face_patch.shape[:2]
        if h < 40 or w < 40:
            print("[Enhance] face too small, skipping")
            return face_patch

        # --------- FAST MODE: только быстрый шарпинг ----------
        if self.mode == "fast" or (self.gfpgan is None and self.esr is None):
            t0 = time.perf_counter()
            out = adaptive_sharpen(face_patch)
            dt = (time.perf_counter() - t0) * 1000
            print(f"[FS] ENHANCE_FAST (sharpen only): {dt:.1f} ms")
            return out

        # --------- QUALITY MODE: полный пайплайн ----------
        t_total0 = time.perf_counter()

        # 1) letterbox → 512
        t0 = time.perf_counter()
        face_512, meta = safe_letterbox_to_512(face_patch)
        print(f"[FS] ENH letterbox: {(time.perf_counter() - t0) * 1000:.1f} ms")

        # 2) GFPGAN
        if self.gfpgan:
            t0 = time.perf_counter()
            gfp_512 = self.gfpgan.enhance_face(face_512)
            print(f"[FS] ENH GFPGAN: {(time.perf_counter() - t0) * 1000:.1f} ms")
        else:
            gfp_512 = face_512

        # 3) ESRGAN ×4 (если включён)
        if self.esr:
            t0 = time.perf_counter()
            upscaled = self.esr.upscale(gfp_512)
            print(f"[FS] ENH ESRGAN: {(time.perf_counter() - t0) * 1000:.1f} ms")
        else:
            upscaled = gfp_512

        # 4) “unletterbox”
        t0 = time.perf_counter()
        top, left, nh, nw = meta
        unboxed = upscaled[top:top + nh, left:left + nw]
        print(f"[FS] ENH unletterbox: {(time.perf_counter() - t0) * 1000:.1f} ms")

        # 5) resize обратно
        t0 = time.perf_counter()
        enhanced_resized = cv2.resize(
            unboxed,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        print(f"[FS] ENH resize_back: {(time.perf_counter() - t0) * 1000:.1f} ms")

        # 6) цвет под оригинальное лицо
        t0 = time.perf_counter()
        matched = color_match_hist(face_patch, enhanced_resized)
        print(f"[FS] ENH color_match: {(time.perf_counter() - t0) * 1000:.1f} ms")

        # 7) адаптивный шарпинг
        t0 = time.perf_counter()
        sharpened = adaptive_sharpen(matched)
        print(f"[FS] ENH sharpen: {(time.perf_counter() - t0) * 1000:.1f} ms")

        out = (
            face_patch.astype(np.float32) * 0.70 +
            sharpened.astype(np.float32) * 0.30
        )
        out = np.clip(out, 0, 255).astype(np.uint8)

        t_total = (time.perf_counter() - t_total0) * 1000
        print(f"[FS] ENHANCE_QUALITY total: {t_total:.1f} ms")
        return out

    # =================================================================

    def swap_face(self, user_photo_bytes, template_bytes):
        """
        Основной публичный метод: байты user → байты template → готовое изображение BGR.
        Логируем время каждого крупного этапа.
        """
        t_all0 = time.perf_counter()
        print("========== [FS] NEW REQUEST ==========")

        # -------- decode --------
        user_img = self._decode(user_photo_bytes, "USER")
        template_img = self._decode(template_bytes, "TEMPLATE")

        # -------- detect faces --------
        src_face = self._main_face(user_img, "USER")
        dst_face = self._main_face(template_img, "TEMPLATE")

        # -------- swap --------
        t0 = time.perf_counter()
        swapped_full = self.swapper.get(
            template_img,
            dst_face,
            src_face,
            paste_back=True,
        )
        if swapped_full is None:
            raise RuntimeError("Face swap failed")
        t_sw = (time.perf_counter() - t0) * 1000
        print(f"[FS] SWAP (InSwapper): {t_sw:.1f} ms")

        # Если нет улучшайзеров вообще — сразу выходим
        if self.mode == "fast" and self.gfpgan is None and self.esr is None:
            t_total = (time.perf_counter() - t_all0) * 1000
            print(f"[FS] BLEND: 0.0 ms (not needed, already pasted)")
            print(f"[FS] TOTAL (fast, no extra enhance): {t_total:.1f} ms")
            return swapped_full

        # -------- enhancement + blend --------
        try:
            x1, y1, x2, y2 = dst_face.bbox.astype(int)

            # safety clamp
            h_full, w_full = swapped_full.shape[:2]
            x1 = max(0, min(x1, w_full - 1))
            x2 = max(1, min(x2, w_full))
            y1 = max(0, min(y1, h_full - 1))
            y2 = max(1, min(y2, h_full))

            face_patch = swapped_full[y1:y2, x1:x2]

            # enhance
            t0 = time.perf_counter()
            enhanced_patch = self._enhance_face_patch(face_patch)
            t_enh = (time.perf_counter() - t0) * 1000
            print(f"[FS] ENHANCE total: {t_enh:.1f} ms")

            ph, pw = enhanced_patch.shape[:2]

            # Маска для alpha-blend
            t0 = time.perf_counter()
            mask = make_ellipse_mask(ph, pw)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]  # H×W×1

            roi = swapped_full[y1:y2, x1:x2].astype(np.float32)
            enh = enhanced_patch.astype(np.float32)
            blended = enh * alpha + roi * (1.0 - alpha)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            swapped_full[y1:y2, x1:x2] = blended
            t_blend = (time.perf_counter() - t0) * 1000
            print(f"[FS] BLEND (alpha): {t_blend:.1f} ms")

        except Exception as e:
            print("[FaceSwap] enhancement/blending error:", e)

        t_total = (time.perf_counter() - t_all0) * 1000
        print(f"[FS] TOTAL request: {t_total:.1f} ms")
        return swapped_full
