from typing import List
import numpy as np


def albedo_map(front_light_images: List[np.ndarray], mask=None) -> np.ndarray:
    """Computes the albedo mapping given the light images.
    Uses the exposure fusion algorithm.
    Mertens, Tom, Jan Kautz, and Frank Van Reeth. "Exposure fusion." 15th Pacific Conference on Computer Graphics and Applications (PG'07). IEEE, 2007.

    Parameters
    ----------
    front_light_images : List[np.ndarray]
        The front light images.
    mask : _type_, optional
        The mask, by default None

    Returns
    -------
    np.ndarray
        The albedo map.
    """
    import cv2 as cv

    merge_mertens = cv.createMergeMertens()
    albedo_map = merge_mertens.process(front_light_images)

    if isinstance(mask, np.ndarray):
        albedo_map[mask == 0] = [0, 0, 0]

    albedo_map = np.clip(albedo_map * 255, 0, 255).astype(np.uint8)

    return albedo_map
