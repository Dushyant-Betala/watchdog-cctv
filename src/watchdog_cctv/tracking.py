"""Ultralytics adapter. Create one instance for each independent video."""

from .types import Observation


class UltralyticsTracker:
    def __init__(
        self, model="yolo11n.pt", tracker="botsort.yaml", device=None, confidence=0.25
    ):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Install video inference with: python -m pip install -e '.[vision]'"
            ) from error
        self.model = YOLO(model)
        self.tracker = tracker
        self.device = device
        self.confidence = confidence
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.kinds = {
            identifier: "person" if name == "person" else "item"
            for identifier, name in names.items()
            if name in {"person", "backpack", "handbag", "suitcase"}
        }
        if "person" not in self.kinds.values() or "item" not in self.kinds.values():
            raise ValueError(
                "The model must include person and at least one backpack, handbag, or suitcase class"
            )

    def track(self, image):
        result = self.model.track(
            image,
            persist=True,
            tracker=self.tracker,
            classes=list(self.kinds),
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return ()
        observations = []
        for track_id, class_id, bbox, confidence in zip(
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.xyxyn.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            coordinates = tuple(max(0.0, min(1.0, value)) for value in bbox)
            if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                continue
            observations.append(
                Observation(
                    int(track_id),
                    self.kinds[int(class_id)],
                    coordinates,
                    float(confidence),
                )
            )
        return tuple(observations)
