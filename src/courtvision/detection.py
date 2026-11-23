"""Object detection (players, ball, hoop, referees) with a Roboflow model.

Runs locally on the Colab GPU via the `inference` package; the API key is only used
to download weights. Set ROBOFLOW_API_KEY (Colab: Secrets panel).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLES = ("player", "ball", "hoop", "referee")


@dataclass
class FrameDetections:
    players: Any   # sv.Detections
    ball: Any
    hoop: Any
    referees: Any


def split_roles(dets, classes: dict[str, Any]) -> FrameDetections:
    """Split one frame's sv.Detections into roles using the configured class names."""
    names = np.asarray(dets.data.get("class_name", np.array([""] * len(dets))))
    picked = {}
    for role in ROLES:
        wanted = classes.get(role) or []
        wanted = [wanted] if isinstance(wanted, str) else wanted
        picked[role] = dets[np.isin(names, wanted)] if len(dets) else dets
    return FrameDetections(players=picked["player"], ball=picked["ball"],
                           hoop=picked["hoop"], referees=picked["referee"])


def prepare_gpu() -> None:
    """Load the CUDA/cuDNN libraries ONNX Runtime needs from the installed NVIDIA packages.

    Without this, onnxruntime-gpu often cannot find them on Colab and silently runs the
    models on the CPU. Safe to call more than once; does nothing without onnxruntime-gpu.
    """
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):  # onnxruntime >= 1.21
            ort.preload_dlls()
    except Exception:  # noqa: BLE001 - best effort; the model then reports CPU fallback
        pass


def gpu_status() -> str:
    """One line describing whether the models can use the GPU."""
    parts = []
    try:
        import torch

        parts.append(f"torch {torch.__version__} CUDA available: {torch.cuda.is_available()}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"torch not usable ({e.__class__.__name__})")
    try:
        import onnxruntime as ort

        prepare_gpu()
        parts.append(f"onnxruntime {ort.__version__} providers: {ort.get_available_providers()}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"onnxruntime not usable ({e.__class__.__name__})")
    return " | ".join(parts)


class Detector:
    def __init__(self, det_config: dict[str, Any]):
        prepare_gpu()
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
        return split_roles(sv.Detections.from_inference(result), self.classes)

    def class_names(self) -> list[str]:
        """Class names the model can output (to check the config against)."""
        names = getattr(self.model, "class_names", None)
        return list(names) if names is not None else []
