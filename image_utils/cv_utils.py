from typing import List, TypeVar

import cv2
import numpy as np

T = TypeVar("+R", bound=int | float)
# NPARRAY = TypeVar("np.array", bound=np.array)


def draw_polygon_on_image(
    image: np.array,
    polygons: List[List[T]],
    is_closed=True,
    color=[0, 0, 255],
    thickness=3,
) -> np.array:
    polygons = [
        np.array(polygon, dtype=np.int32).reshape(-1, 2) for polygon in polygons
    ]
    for coords in polygons:
        image = cv2.polylines(image, [coords], is_closed, color, thickness)

    return image
