import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_detrac_annotations(xml_path: Path):
    """
    Parse a UA-DETRAC XML file into a dict:
    { frame_num (int): [ { 'id': str, 'bbox': [x, y, w, h], 'class': str }, ... ] }
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    annotations = {}

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))

        target_list = frame.find("target_list")
        if target_list is None:
            annotations[frame_num] = []
            continue

        targets = []
        for target in target_list.findall("target"):
            tid = target.get("id")

            box = target.find("box")
            attr = target.find("attribute")
            if box is None or attr is None:
                continue

            left = float(box.get("left"))
            top = float(box.get("top"))
            width = float(box.get("width"))
            height = float(box.get("height"))
            vehicle_class = attr.get("vehicle_type")

            targets.append({
                "id": tid,
                "bbox": [left, top, width, height],
                "class": vehicle_class,
            })

        annotations[frame_num] = targets

    return annotations


def predict_vehicle_class(crop_rgb_np, model, idx_to_class, device="cpu"):
    """
    Predicts the class and confidence for a cropped vehicle image.
    
    Args:
        crop_rgb_np: numpy array (H, W, 3), either [0,1] or [0,255].
        model: PyTorch model.
        idx_to_class: dict mapping class index -> class string.
        device: "cpu" or "cuda".

    Returns:
        (label_str, confidence_float)
    """
    # Ensure float32
    if crop_rgb_np.dtype != "float32":
        crop = crop_rgb_np.astype("float32")
    else:
        crop = crop_rgb_np.copy()

    # Ensure in range [0,1]
    if crop.max() > 1.0:
        crop /= 255.0

    # Convert to PyTorch tensor (1,3,H,W)
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())
    pred_label = idx_to_class[pred_idx]

    return pred_label, conf


class VehicleClassifier(nn.Module):
    """
    Simple CNN for vehicle classification on 64x64 RGB crops.

    If you change input resolution, you may need to adjust the
    Linear layer's input size in `self.classifier`.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),  # for 64x64 input -> 4x4 after 4 pools
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
