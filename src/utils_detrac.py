import xml.etree.ElementTree as ET
from pathlib import Path

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
