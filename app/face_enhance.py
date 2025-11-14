import cv2
import numpy as np
import onnxruntime as ort


class FaceEnhancer:
    def __init__(self, model_path: str):
        print("[Enhancer] Loading GFPGAN ONNX...")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def enhance_face(self, face_bgr: np.ndarray) -> np.ndarray:
        # BGR → RGB
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = rgb.astype(np.float32) / 255.0
        t = np.transpose(t, (2, 0, 1))
        t = np.expand_dims(t, 0)

        output = self.session.run(None, {self.input_name: t})[0][0]

        # back to BGR uint8
        out = np.transpose(output, (1, 2, 0))
        out = np.clip(out * 255.0, 0, 255).astype("uint8")
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
