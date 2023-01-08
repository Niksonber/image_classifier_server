import numpy as np
import onnxruntime as rt
from PIL import Image

from constants import IMAGENET_INDEX, PROVIDERS, ONNX_MODEL_PATH


class ImageClassifier:
    __session = rt.InferenceSession(str(ONNX_MODEL_PATH), providers=PROVIDERS)

    @staticmethod
    def pre_process(img: Image):
        """Resize (224, 224), zero center and Convert RGB to BGR"""
        X = np.array(img.resize((224, 224)), dtype=np.float32) - 128
        return np.expand_dims(X[:, :, ::-1], axis=0)

    @staticmethod
    def decode_predictions(pred: np.ndarray, top_k: int):
        def get_label(class_, pred):
            return (*IMAGENET_INDEX[class_], str(round(pred[class_], 3)))

        pred = np.array(pred).squeeze()
        sorted_pred = np.argsort(pred, axis=-1)[-top_k:][::-1]
        return [get_label(class_, pred) for class_ in sorted_pred]

    @classmethod
    def predict(cls, img: Image, top_k: int = 3):
        pred = cls.__session.run(['predictions'], {"input": cls.pre_process(img)})
        return cls.decode_predictions(pred, top_k)
