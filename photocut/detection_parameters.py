"""Immutable detection parameter value object for reproducible evaluation."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class DetectionParameters:
    ps_highlight: int = 230
    gaussian_blur_ksize: int = 5
    canny_low: int = 30
    canny_high: int = 90
    hough_threshold: int = 35
    hough_min_line_length: int = 40
    hough_max_line_gap: int = 25
    horizontal_angle_tolerance: float = 10.0
    vertical_angle_tolerance: float = 10.0
    weight_proximity: float = 0.5
    weight_length: float = 0.3
    weight_angle: float = 0.2
    confidence_threshold: float = 0.6
    corner_confidence_threshold: float = 0.7
    roi_scales: tuple = (0.10, 0.15, 0.20, 0.25)

    def __post_init__(self):
        if not 1 <= self.ps_highlight <= 255:
            raise ValueError("ps_highlight must be in [1, 255]")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("canny thresholds must satisfy 0 <= low < high <= 255")
        if self.gaussian_blur_ksize <= 0 or self.gaussian_blur_ksize % 2 == 0:
            raise ValueError("gaussian_blur_ksize must be a positive odd integer")
        if any(value <= 0 for value in (
            self.hough_threshold,
            self.hough_min_line_length,
            self.hough_max_line_gap,
        )):
            raise ValueError("Hough parameters must be positive")
        weights = (self.weight_proximity, self.weight_length, self.weight_angle)
        if any(value < 0 for value in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("scoring weights must be non-negative and sum to 1")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0 <= self.corner_confidence_threshold <= 1:
            raise ValueError("corner_confidence_threshold must be in [0, 1]")
        if not self.roi_scales or any(not 0 < value <= 0.5 for value in self.roi_scales):
            raise ValueError("roi_scales must be in (0, 0.5]")
        if tuple(sorted(set(self.roi_scales))) != self.roi_scales:
            raise ValueError("roi_scales must be unique and ascending")
        if not 0 <= self.horizontal_angle_tolerance <= 90:
            raise ValueError("horizontal_angle_tolerance must be in [0, 90]")
        if not 0 <= self.vertical_angle_tolerance <= 90:
            raise ValueError("vertical_angle_tolerance must be in [0, 90]")

    def replace(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping):
            raise ValueError("detection parameters must be an object")
        data = dict(value)
        if "roi_scales" in data:
            raw_scales = data["roi_scales"]
            if not isinstance(raw_scales, (list, tuple)):
                raise ValueError("roi_scales must be a list or tuple")
            data["roi_scales"] = tuple(raw_scales)
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError("invalid detection parameter fields") from exc

    def to_dict(self):
        value = asdict(self)
        value["roi_scales"] = list(self.roi_scales)
        return value


DEFAULT_DETECTION_PARAMETERS = DetectionParameters()
