import cv2
import numpy as np
import onnxruntime as ort


def _get_onnx_providers():
    available = ort.get_available_providers()
    providers = []
    if "OpenVINOExecutionProvider" in available:
        providers.append("OpenVINOExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def _create_onnx_session(onnx_path: str) -> ort.InferenceSession:
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
        providers=_get_onnx_providers()
    )


class FaceEnhancer:
    """
    Wrapper over GFPGANv1.4.onnx (CPU/OpenVINO)
    Requires input exactly (512,512,3)
    """

    def __init__(self, model_path: str):
        self.sess = _create_onnx_session(model_path)
        self.input_name = self.sess.get_inputs()[0].name

    def enhance_face(self, img512):
        """
        img512 must be exactly (512,512,3)
        fallback-safe: returns input if anything is off
        """

        if img512 is None:
            print("[GFPGAN] Input is None")
            return img512

        if img512.shape != (512, 512, 3):
            print(f"[GFPGAN] Invalid shape {img512.shape}, expected (512,512,3)")
            return img512

        try:
            face = cv2.cvtColor(img512, cv2.COLOR_BGR2RGB)
            face = face.astype(np.float32) / 255.0
            face = np.transpose(face, (2, 0, 1))  # CHW
            face = np.expand_dims(face, 0)        # 1x3x512x512

            out = self.sess.run(None, {self.input_name: face})[0]

            out = np.squeeze(out)
            out = np.transpose(out, (1, 2, 0))
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            return out

        except Exception as e:
            print("[GFPGAN] ERROR:", e)
            return img512
