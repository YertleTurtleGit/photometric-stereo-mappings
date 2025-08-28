from typing import List
import cv2 as cv
import numpy as np


def translucency_map(back_light_images: List[np.ndarray], mask=None) -> np.ndarray:
    """Computes the translucency mapping given the back light images.
    Uses the exposure fusion algorithm.
    Mertens, Tom, Jan Kautz, and Frank Van Reeth. "Exposure fusion." 15th Pacific Conference on Computer Graphics and Applications (PG'07). IEEE, 2007.

    Parameters
    ----------
    back_light_images : List[np.ndarray]
        The back light images.
    mask : _type_, optional
        The mask, by default None

    Returns
    -------
    np.ndarray
        The translucency map.
    """
    import cv2 as cv

    merge_mertens = cv.createMergeMertens()
    translucency_map_image = merge_mertens.process(back_light_images)

    if isinstance(mask, np.ndarray):
        translucency_map_image[mask == 0] = [0, 0, 0]

    translucency_map_image = np.clip(translucency_map_image * 255, 0, 255).astype(
        np.uint8
    )

    return translucency_map_image
