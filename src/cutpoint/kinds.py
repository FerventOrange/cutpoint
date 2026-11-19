"""What ``cutpoint edit --kind`` can make (:data:`KINDS`), apart from :mod:`cutpoint.edit` so
that the command-line parser offers the kinds without loading the export modules."""

from __future__ import annotations

from dataclasses import dataclass

from cutpoint.styles import AUTO, FUNNY, HYPE


@dataclass(frozen=True)
class Kind:
    """What one ``--kind`` makes.

    Attributes:
        style: Analysis style (``--style`` overrides it); :data:`~cutpoint.styles.AUTO`:
            the style is detected from the footage; an ambiguous detection uses its best
            guess's style if that is gameplay, else the General style
            (:func:`~cutpoint.styles.auto_fallback`).
        top: Clip the N best moments (``clips --top``); None = no count limit.
        max_total: Cap the clips' total length in seconds (``clips --max-total``).
        order: Clip numbering and playback order (``clips --order``).
        transition: ``assemble --transition``.
        loudnorm: ``assemble --loudnorm``.
        captions: ``assemble --captions`` (burned in; whisper runs on the final).
        emphasis: ``--style auto``'s emphasis (:class:`~cutpoint.styles.AutoRequest`):
            ``funny`` for funny-moments, else ``hype``.
    """

    style: str
    top: int | None = None
    max_total: float | None = None
    order: str = "score"
    transition: str = "cut"
    loudnorm: bool = True
    captions: bool = False
    emphasis: str = HYPE


KINDS: dict[str, Kind] = {
    "highlights": Kind(AUTO, max_total=180, order="chronological", transition="fade"),
    "montage": Kind(AUTO, max_total=60, order="score", transition="fade"),
    "clips": Kind(AUTO, top=10, order="score", transition="cut"),
    "funny-moments": Kind(
        "funny", top=8, order="chronological", transition="cut", captions=True, emphasis=FUNNY
    ),
}
