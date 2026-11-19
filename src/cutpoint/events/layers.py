"""Names of the ``cutpoint events`` layers, without imports, so the command-line parser can
offer them without loading the events pipeline."""

LAYERS = ("audio", "voice", "visual", "crash", "ocr", "vlm")
# Layers run by default; voice/visual/crash feed action-cam features (other styles opt in).
DEFAULT_LAYERS = ("audio", "ocr", "vlm")
ACTION_CAM_LAYERS = ("audio", "voice", "visual", "crash", "vlm")
