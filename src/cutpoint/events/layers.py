"""Names of the ``cutpoint events`` layers, without imports, so the command-line parser can
offer them without loading the events pipeline."""

LAYERS = ("audio", "voice", "visual", "crash", "ocr", "overlay", "vlm")
# Layers run by default; voice/visual/crash feed action-cam features (other styles opt in).
# overlay: the on-screen overlays of non-gameplay footage (:mod:`cutpoint.events.overlays`),
# nothing for gameplay styles (:func:`overlay_kinds`)
DEFAULT_LAYERS = ("audio", "ocr", "overlay", "vlm")
ACTION_CAM_LAYERS = ("audio", "voice", "visual", "crash", "overlay", "vlm")

# The overlays the overlay layer looks for (#1069 (c)), per kind the project styles (a name or
# a learned style's base) whose moments its events add points to: a scoreboard (sport, and
# action footage, which general footage gets from ``--style auto``), gauges such as speed or
# altitude (action footage) and lower-thirds (talk). No kind for the gameplay styles: their
# HUD is the OCR layer's. Removing an entry switches that kind off as a unit (its event type
# then never gets an event: :data:`cutpoint.events.eventtypes.OVERLAY_TYPES`).
OVERLAY_STYLES: dict[str, tuple[str, ...]] = {
    "scoreboard": ("action", "sport"),
    "gauge": ("action", "sport", "action-cam"),
    "lower_third": ("commentary",),
}


def overlay_kinds(*names: str) -> tuple[str, ...]:
    """The overlay kinds (:data:`OVERLAY_STYLES`) for a project whose style has one of
    ``names`` (its name and base); empty for any other style."""
    return tuple(k for k, styles in OVERLAY_STYLES.items() if any(n in styles for n in names))
