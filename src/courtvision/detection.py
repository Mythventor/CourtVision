"""Object detection (players, ball, hoop, referees) with a Roboflow model.

Runs locally on the Colab GPU via the `inference` package; the API key is only used
to download weights. Set ROBOFLOW_API_KEY (Colab: Secrets panel).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FrameDetections:
    players: Any   # sv.Detections
    ball: Any
    hoop: Any
    referees: Any


class Detector:
    def __init__(self, det_config: dict[str, Any]):
        from inference import get_model

        model_id = det_config["model_id"]
        if not model_id or model_id == "TODO":
            raise ValueError("Set detection.model_id in the config to a Roboflow model id.")
        self.model = get_model(model_id=model_id, api_key=os.environ.get("ROBOFLOW_API_KEY"))
        self.confidence = det_config.get("confidence", 0.3)
        self.classes = det_config["classes"]

    def __call__(self, frame: np.ndarray) -> FrameDetections:
        import supervision as sv

        result = self.model.infer(frame, confidence=self.confidence)[0]
        dets = sv.Detections.from_inference(result)
        names = dets.data.get("class_name", np.array([]))

        def pick(key: str) -> sv.Detections:
            return dets[names == self.classes[key]]

        return FrameDetections(
            players=pick("player"), ball=pick("ball"), hoop=pick("hoop"), referees=pick("referee")
        )
