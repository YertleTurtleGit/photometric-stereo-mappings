import numpy as np
from typing import Tuple


def identifier_map(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    import cv2 as cv
    from random import randint

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    element_count = len(contours)
    identifier_map_image = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

    color_indices = set()
    while len(color_indices) < element_count:
        color_indices.add((randint(50, 255), randint(50, 255), randint(50, 255)))

    for index in range(element_count):
        identifier_map_image = cv.drawContours(
            identifier_map_image, [contours[index]], 0, color_indices.pop(), cv.FILLED
        )

    return identifier_map_image, element_count
