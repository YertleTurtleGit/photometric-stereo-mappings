import numpy as np
from utilities.process_logger import ProcessLogger


def cavity_map(
    height_map: np.ndarray, mask: np.ndarray, logger: ProcessLogger = ProcessLogger()
) -> np.ndarray:
    """Calculates the ambient occlusion map.

    Parameters
    ----------
    height_map : np.ndarray
        The input height map.
    mask : np.ndarray
        The input mask.
    logger : ProcessLogger, optional (default=ProcessLogger()).
        The logger.

    Returns
    -------
    np.ndarray
        The ambient occlusion map.
    """
    from DEFINITIONS import CAVITY_MAP_FACTOR, CAVITY_MAP_SUMMAND
    from math import sqrt
    from threading import Thread, Lock
    from typing import List

    NEIGHBORHOOD_RADIUS: int = 3
    neighborhood_area: int = (NEIGHBORHOOD_RADIUS * 2 + 1) ** 2
    height_map = height_map.astype(np.float64) / 255

    ambient_occlusion_map = np.zeros((height_map.shape[0], height_map.shape[1]))
    ambient_occlusion_lock: Lock = Lock()
    ambient_occlusion_threads: List[Thread] = []

    def calculate_ambient_occlusion_kernel_point(x: int, y: int, weight: float) -> None:
        ambient_occlusion_lock.acquire()
        neighbor_height_map = np.roll(height_map, (x, y), axis=(0, 1))
        ambient_occlusion_lock.release()

        height_delta_map = (
            (height_map - neighbor_height_map) / neighborhood_area
        ) * weight

        ambient_occlusion_lock.acquire()
        nonlocal ambient_occlusion_map
        ambient_occlusion_map += height_delta_map
        ambient_occlusion_lock.release()

    maximum_weight = sqrt(
        (NEIGHBORHOOD_RADIUS + 1) ** 2 + (NEIGHBORHOOD_RADIUS + 1) ** 2
    )

    for x in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
        for y in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            if x == 0 and y == 0:
                continue

            distance: float = sqrt(x**2 + y**2)
            weight: float = maximum_weight - distance

            if distance > NEIGHBORHOOD_RADIUS:
                continue

            thread = Thread(
                target=calculate_ambient_occlusion_kernel_point,
                args=(
                    x,
                    y,
                    weight,
                ),
            )
            ambient_occlusion_threads.append(thread)
            thread.start()

            while len(ambient_occlusion_threads) >= 8:  # TODO 8 is a magic number
                for wait_thread in ambient_occlusion_threads:
                    wait_thread.join()
                    ambient_occlusion_threads.remove(wait_thread)
                    break

    for thread in ambient_occlusion_threads:
        thread.join()

    ambient_occlusion_map[mask > 0] += CAVITY_MAP_SUMMAND
    ambient_occlusion_map[mask > 0] *= CAVITY_MAP_FACTOR

    ambient_occlusion_map *= 255

    if (
        np.max(ambient_occlusion_map[mask > 0]) > 255
        or np.min(ambient_occlusion_map[mask > 0]) < 0
    ):
        logger.warn(
            "Ambient occlusion map values are getting clipped. "
            + "Consider changing the summand or factor. (min: "
            + str(np.min(ambient_occlusion_map[mask > 0]))
            + ", max: "
            + str(np.max(ambient_occlusion_map[mask > 0]))
            + ", median: "
            + str(np.median(ambient_occlusion_map[mask > 0]))
            + ")"
        )

    ambient_occlusion_map = ambient_occlusion_map.astype(np.uint8)
    ambient_occlusion_map[mask == 0] = 0

    return ambient_occlusion_map
