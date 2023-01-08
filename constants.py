import json
from pathlib import Path


CURRENT_FOLDER = Path(__file__).parent
RESOURCES_FOLDER = CURRENT_FOLDER / 'resources'

IMAGENET_INDEX = json.load((RESOURCES_FOLDER / 'imagenet_class_index.json').open())
ONNX_MODEL_PATH = RESOURCES_FOLDER / 'resnet50.onnx'
PROVIDERS = ['CPUExecutionProvider']
