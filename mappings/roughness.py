import logging
import numpy as np


def roughness_map(
    normal_map: np.ndarray, mask: np.ndarray, maximum_thread_count: int = 4
) -> np.ndarray:
    """Calculates the roughness map.

    Parameters
    ----------
    normal_map : np.ndarray
        The input normal map.
    mask : np.ndarray
        The input mask.
    maximum_thread_count : int, optional (default=4).
        The maximum thread count.


    Returns
    -------
    np.ndarray
        The roughness map.
    """

    import cv2 as cv
    from math import sqrt
    from threading import Thread, Lock
    from typing import List

    try:
        from DEFINITIONS import (
            ROUGHNESS_MAP_FACTOR,
            ROUGHNESS_MAP_SUMMAND,
        )
        from mappings.normal import decode_normal_map_image
    except ModuleNotFoundError:
        from photometric_stereo_mappings.DEFINITIONS import (
            ROUGHNESS_MAP_FACTOR,
            ROUGHNESS_MAP_SUMMAND,
        )
        from photometric_stereo_mappings.mappings.normal import decode_normal_map_image

    NEIGHBORHOOD_RADIUS: int = 1
    ANGLE_DELTA_UPPER_CLIP: float = np.pi * 0.025
    MEDIAN_BLUR_RADIUS: int = 1

    normal_map[mask == 0] = [0, 0, 0]
    normal_map = decode_normal_map_image(normal_map)

    roughness_map = np.zeros((normal_map.shape[0], normal_map.shape[1]))
    roughness_lock: Lock = Lock()
    roughness_threads: List[Thread] = []

    def calculate_roughness_kernel_point(x: int, y: int, weight: float) -> None:
        roughness_lock.acquire()
        neighbor_normal_map = np.roll(normal_map, (x, y), axis=(0, 1))
        roughness_lock.release()

        dot_product_map = (
            normal_map[:, :, 0] * neighbor_normal_map[:, :, 0]
            + normal_map[:, :, 1] * neighbor_normal_map[:, :, 1]
            + normal_map[:, :, 2] * neighbor_normal_map[:, :, 2]
        )

        dot_product_map = np.clip(dot_product_map, -1, 1)
        angle_delta_map = np.absolute(np.arccos(dot_product_map))
        angle_delta_map = np.clip(angle_delta_map, 0, ANGLE_DELTA_UPPER_CLIP)
        angle_delta_map *= weight

        roughness_lock.acquire()
        nonlocal roughness_map
        roughness_map += angle_delta_map
        roughness_lock.release()

    maximum_distance: float = sqrt(((NEIGHBORHOOD_RADIUS + 1) ** 2) * 2)

    for x in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
        for y in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            if x == 0 and y == 0:
                continue

            distance: float = sqrt(x**2 + y**2)
            weight: float = maximum_distance - distance

            if distance > NEIGHBORHOOD_RADIUS:
                continue

            thread = Thread(
                target=calculate_roughness_kernel_point,
                args=(x, y, weight),
            )
            roughness_threads.append(thread)
            thread.start()

            while len(roughness_threads) >= maximum_thread_count:
                for wait_thread in roughness_threads:
                    wait_thread.join()
                    roughness_threads.remove(wait_thread)
                    break

    for thread in roughness_threads:
        thread.join()

    roughness_map += ROUGHNESS_MAP_SUMMAND
    roughness_map *= ROUGHNESS_MAP_FACTOR
    roughness_map *= 255

    if np.max(roughness_map[mask > 0]) > 255 or np.min(roughness_map[mask > 0]) < 0:
        logging.warn(
            "Roughness map values are getting clipped. "
            + "Consider changing the summand or factor. (min: "
            + str(np.min(roughness_map[mask > 0]))
            + ", max: "
            + str(np.max(roughness_map[mask > 0]))
            + ", median: "
            + str(np.median(roughness_map[mask > 0]))
            + ")"
        )
        roughness_map = np.clip(roughness_map, 0, 255)

    roughness_map = roughness_map.astype(np.uint8)
    blurred_roughness_map = cv.GaussianBlur(roughness_map, (3, 3), 0)
    roughness_map = cv.addWeighted(blurred_roughness_map, 1.5, roughness_map, -0.5, 0)
    roughness_map = cv.medianBlur(roughness_map, MEDIAN_BLUR_RADIUS * 2 + 1)
    roughness_map[mask == 0] = 0

    return roughness_map
