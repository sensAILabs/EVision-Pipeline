"""Drop-in Stage 2 EV vs ICE classifier for consolidated_dashcam_pipeline.ipynb.

Trained by training_pipeline.ipynb. Matches the ``score_crop`` interface of
``LocalYOLOEVClassifier`` so the dashcam pipeline can swap models with minimal changes.

Supports both:
- torchvision checkpoints (state_dict + backbone name in metadata)
- Ultralytics YOLO-cls checkpoints (full ``best.pt``, family ``yolo_cls``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torchvision import models, transforms  # pyright: ignore[reportMissingImports]


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_torchvision_backbone(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    weights = "DEFAULT"

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == "vit_b_32":
        model = models.vit_b_32(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported torchvision backbone: {model_name}")
    return model


@dataclass
class LocalEVClassifier:
    """Binary EV vs ICE classifier loaded from training_pipeline exports."""

    model_dir: str | Path
    confidence_threshold: float = 0.70
    device: torch.device = field(default_factory=_default_device)
    model: nn.Module | object | None = field(default=None, repr=False)
    transform: transforms.Compose | None = field(default=None, repr=False)
    class_to_idx: dict[str, int] = field(default_factory=dict)
    ev_index: int = 0
    family: str = "torchvision"
    backbone: str = ""
    img_size: int = 224

    def __post_init__(self) -> None:
        model_dir = Path(self.model_dir)
        metadata_path = model_dir / "metadata.json"
        weights_path = model_dir / "best.pt"

        if not metadata_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Expected metadata.json and best.pt under {model_dir}. "
                "Run training_pipeline.ipynb export cell first."
            )

        metadata = json.loads(metadata_path.read_text())
        self.class_to_idx = {str(k): int(v) for k, v in metadata["class_to_idx"].items()}
        self.ev_index = int(metadata["ev_index"])
        self.img_size = int(metadata.get("img_size", 224))
        self.backbone = str(metadata["backbone"])
        self.family = str(metadata.get("family", "torchvision")).lower()

        if self.family == "yolo_cls":
            self._load_yolo_cls(weights_path)
        elif self.family == "torchvision":
            self._load_torchvision(weights_path)
        else:
            raise ValueError(
                f"Unsupported export family '{self.family}'. "
                "Expected 'torchvision' or 'yolo_cls'."
            )

        print(
            f"Loaded EV classifier ({self.family} / {self.backbone}) from {model_dir} "
            f"[metric={metadata.get('selected_by_metric')}, "
            f"experiment={metadata.get('exported_experiment')}]"
        )

    def _load_torchvision(self, weights_path: Path) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model = _build_torchvision_backbone(self.backbone, len(self.class_to_idx))
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _load_yolo_cls(self, weights_path: Path) -> None:
        from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

        self.transform = None
        self.model = YOLO(str(weights_path))
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            name_to_idx = {str(v).lower(): int(k) for k, v in names.items()}
            if "ev" in name_to_idx:
                self.ev_index = name_to_idx["ev"]
            self.class_to_idx = {
                str(v).lower(): int(k) for k, v in names.items()
            }

    def score_crop(self, crop_bgr: np.ndarray) -> tuple[float, str | None]:
        """Return (ev_confidence, ev_label) for a BGR vehicle crop."""
        if self.model is None:
            raise RuntimeError("Classifier is not loaded.")

        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0, None

        if self.family == "yolo_cls":
            return self._score_yolo_cls(crop_bgr)
        return self._score_torchvision(crop_bgr)

    def _score_torchvision(self, crop_bgr: np.ndarray) -> tuple[float, str | None]:
        if self.transform is None or not isinstance(self.model, nn.Module):
            raise RuntimeError("Torchvision classifier is not loaded.")

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        ev_conf = float(probs[self.ev_index].item())
        if ev_conf < self.confidence_threshold:
            return 0.0, None
        return ev_conf, "ev"

    def _score_yolo_cls(self, crop_bgr: np.ndarray) -> tuple[float, str | None]:
        result = self.model.predict(crop_bgr, verbose=False)[0]  # type: ignore[union-attr]
        probs = result.probs
        if probs is None:
            return 0.0, None

        if hasattr(probs, "data"):
            prob_vec = probs.data.detach().cpu().numpy()
        else:
            prob_vec = np.asarray(probs)

        ev_conf = float(prob_vec[self.ev_index])
        if ev_conf < self.confidence_threshold:
            return 0.0, None
        return ev_conf, "ev"
