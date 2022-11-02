from typing import List, TypeVar, Tuple

import cv2
import numpy as np

import image_manipulations_registry

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


def get_masks_from_polygon(
    image_dimensions: Tuple, polygons: List[List[T]]
) -> List[np.array]:
    height, width = image_dimensions

    masks = [np.zeros((height, width), dtype=np.uint8) for i in range(len(polygons))]
    polygons = [
        np.array(polygon, dtype=np.int32).reshape(-1, 2) for polygon in polygons
    ]
    masks_polygons = zip(masks, polygons)
    masks = [
        cv2.fillPoly(mask, [polygon], color=1) for (mask, polygon) in masks_polygons
    ]
    return masks


def draw_text_on_image(image: np.array, string: List, **kwargs) -> np.array:

    image = kwargs.pop("image")
    string = kwargs.pop("string")
    font = kwargs.pop("font", cv2.FONT_HERSHEY_SIMPLEX)
    fontScale = kwargs.pop("fontScale", 0.5)
    fontColor = kwargs.pop("fontColor", (0, 255, 0))
    thickness = kwargs.pop("thickness", 1)
    lineType = kwargs.pop("lineType", 1)
    height, width = image.shape[0:2]
    image = cv2.putText(
        image,
        string,
        (width // 2, height // 2),
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )
    return image


@image_manipulations_registry.IMAGE_MANIPULATIONS.register("contrast_and_brightness")
def change_contrast_and_brightness(
    image: np.array, contrast_control: float, brightness_control: float
) -> np.array:
    # alpha is the contrast control (1.0-3.0)
    # beta is the brightness control (0-100)
    image = cv2.convertScaleAbs(image, alpha=contrast_control, beta=brightness_control)
    return image


@image_manipulations_registry.IMAGE_MANIPULATIONS.register("unsharp_image")
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    # https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


@image_manipulations_registry.IMAGE_MANIPULATIONS.register("resize_image")
def resize_image(
    image: np.array, width: int, height: int, abs_or_frac: str = "frac"
) -> np.array:
    if abs_or_frac == "abs":
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(
            image, None, fx=width, fy=height, interpolation=cv2.INTER_CUBIC
        )
    return image


def make_manipulations_sequencer(manipulations_sequence: List):
    sequence_of_callables = []
    for manipulation in manipulations_sequence:
        sequence_of_callables.append(
            image_manipulations_registry.IMAGE_MANIPULATIONS[manipulation]
        )
    return sequence_of_callables
