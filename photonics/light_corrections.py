import numpy as np
from typing import List


def correct_light_falloff(
    light_images: List[np.ndarray],
    light_falloff_images: List[np.ndarray],
) -> List[np.ndarray]:
    """Corrects light falloff in light images.

    Parameters
    ----------
    light_images : List[np.ndarray]
        The input light images.
    light_falloff_images : List[np.ndarray]
        The input light falloff images.

    Returns
    -------
    List[np.ndarray]
        The corrected images.
    """

    from cv2 import cvtColor, resize, COLOR_GRAY2RGB

    for index, light_image in enumerate(light_images):
        light_image = light_image.astype(np.float64)
        light_falloff_image = light_falloff_images[index]
        light_falloff_image = resize(
            light_falloff_image, (light_image.shape[1], light_image.shape[0])
        )

        if len(light_falloff_image.shape) == 2:
            light_falloff_image = (
                cvtColor(light_falloff_image, COLOR_GRAY2RGB).astype(np.float64) / 255
            )

        light_falloff_image = light_falloff_image.astype(np.float64)
        light_falloff_image -= np.min(light_falloff_image)
        light_falloff_image /= np.max(light_falloff_image)
        light_falloff_image -= 1 / 2
        light_falloff_image = np.clip(light_falloff_image, 0, 1)

        light_image *= 1 - light_falloff_image
        light_image *= 1.5
        light_image = light_image.clip(0, 255).astype(np.uint8)
        light_images[index] = light_image

    return light_images
