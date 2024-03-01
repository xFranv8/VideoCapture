import argparse
from typing import List, Optional, Union

import numpy as np
import torch

import norfair
from norfair import Detection, Tracker, Video

import cv2

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


class YOLO:
    def __init__(self, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument(
    "--img-size", type=int, default="720", help="YOLOv7 inference size (pixels)"
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default="0.5",
    help="YOLOv7 object confidence threshold",
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.7", help="YOLOv7 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track-points",
    type=str,
    default="centroid",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

model = YOLO(device=args.device)

video = Video(0)

distance_function = "iou" if args.track_points == "bbox" else "euclidean"

distance_threshold = (
    DISTANCE_THRESHOLD_BBOX
    if args.track_points == "bbox"
    else DISTANCE_THRESHOLD_CENTROID
)

tracker = Tracker(
    distance_function=distance_function,
    distance_threshold=distance_threshold,
)

for frame in video:
    yolo_detections = model(
        frame,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.img_size,
        classes=args.classes,
    )

    for index, row in yolo_detections.pandas().xyxy[0].iterrows():
        label: str = str(row["class"])
        x1: int = int(row["xmin"])
        y1: int = int(row["ymin"])
        x2: int = int(row["xmax"])
        y2: int = int(row["ymax"])

        """Write label in the frame"""

        if label == "2":
            label = "Car"
        elif label == "3":
            label = "Moto"
        elif label == "5":
            label = "Bus"
        elif label == "7":
            label = "Truck"
        elif label == "0":
            label = "Person"

        cv2.putText(frame, label, (x1 + 25, y1 + 25), cv2.QT_FONT_NORMAL, 0.7, (170, 220, 12), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    detections = yolo_detections_to_norfair_detections(
        yolo_detections, track_points=args.track_points
    )
    tracked_objects = tracker.update(detections=detections)

    if args.track_points == "centroid":
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif args.track_points == "bbox":
        norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_boxes(frame, tracked_objects)

    video.write(frame)

    cv2.imshow("", frame)
    cv2.waitKey(1)
