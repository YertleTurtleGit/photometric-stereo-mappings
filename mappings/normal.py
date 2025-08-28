from typing import List
import numpy as np

try:
    from DEFINITIONS import NORMAL_OPEN_GL
except ModuleNotFoundError:
    from photometric_stereo_mappings.DEFINITIONS import NORMAL_OPEN_GL


def normal_map(
    light_images: List[np.ndarray],
    light_polar_angle: float = 45,
    first_azimuthal_angle: float = 225,
    clockwise: bool = True,
    open_gl: bool = NORMAL_OPEN_GL,
    maximum_dimension: int = 500,
) -> np.ndarray:
    """Computes and saves the normal mapping. Photometric stereo normal mapping.
    Woodham, Robert J. "Photometric method for determining surface orientation from multiple images."
    Optical engineering 19.1 (1980): 139-144.

    Parameters
    ----------
    light_images : List[np.ndarray]
        List of light images.
    light_polar_angle : float, optional
        Polar angle of the light source in degrees. Default is 45.
    first_azimuthal_angle : float, optional
        Azimuthal angle of the light source in degrees. Default is 225.
    open_gl : bool, optional
        If True, the normal map is saved in Open GL format. Default is False.

    Returns
    -------
    np.ndarray
        The normal map.
    """
    if (
        light_images[0].shape[0] > maximum_dimension
        or light_images[0].shape[1] > maximum_dimension
    ):
        light_images_a: List[np.ndarray] = []
        light_images_b: List[np.ndarray] = []

        for light_image in light_images:
            if light_image.shape[0] > light_image.shape[1]:
                light_images_a.append(light_image[: light_image.shape[0] // 2, :, :])
                light_images_b.append(light_image[light_image.shape[0] // 2 :, :, :])
            else:
                light_images_a.append(light_image[:, : light_image.shape[1] // 2, :])
                light_images_b.append(light_image[:, light_image.shape[1] // 2 :, :])

        normal_map_image_a = normal_map(
            light_images_a,
            light_polar_angle,
            first_azimuthal_angle,
            open_gl,
            maximum_dimension,
        )
        normal_map_image_b = normal_map(
            light_images_b,
            light_polar_angle,
            first_azimuthal_angle,
            open_gl,
            maximum_dimension,
        )

        normal_map_image = np.zeros(light_images[0].shape, dtype=np.uint8)

        if light_images[0].shape[0] > light_images[0].shape[1]:
            normal_map_image[: light_images[0].shape[0] // 2, :, :] = normal_map_image_a
            normal_map_image[light_images[0].shape[0] // 2 :, :, :] = normal_map_image_b
        else:
            normal_map_image[:, : light_images[0].shape[1] // 2, :] = normal_map_image_a
            normal_map_image[:, light_images[0].shape[1] // 2 :, :] = normal_map_image_b

        return normal_map_image

    from math import sin, cos, radians

    LIGHT_COUNT: int = len(light_images)
    light_positions = []

    for i in range(LIGHT_COUNT):
        light_azimuthal_angle = first_azimuthal_angle
        step = i * (360 / LIGHT_COUNT)
        if clockwise:
            light_azimuthal_angle -= step
        else:
            light_azimuthal_angle += step

        while light_azimuthal_angle < 0:
            light_azimuthal_angle += 360
        while light_azimuthal_angle >= 360:
            light_azimuthal_angle -= 360

        x = cos(radians(light_azimuthal_angle)) * sin(radians(light_polar_angle))
        y = sin(radians(light_azimuthal_angle)) * sin(radians(light_polar_angle))
        z = cos(radians(light_polar_angle))

        light_positions.append([x, y, z])

    light_matrix = np.array(light_positions).T
    luminosity_matrix: np.ndarray = None
    height = 0
    width = 0

    for i in range(len(light_images)):
        image = light_images[i].astype(np.float64)
        image = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
        image /= 255

        if luminosity_matrix is None:
            height, width = image.shape
            luminosity_matrix = image.reshape((-1, 1))
        else:
            luminosity_matrix = np.append(
                luminosity_matrix, image.reshape((-1, 1)), axis=1
            )

    transposed_luminosity_matrix = luminosity_matrix.T
    normal_matrix = np.linalg.lstsq(
        light_matrix.T, transposed_luminosity_matrix, rcond=None
    )[0].T

    if not open_gl:
        normal_matrix[:, 1] *= -1

    normal_map_image = np.reshape(normal_matrix, (luminosity_matrix.shape[0], 3))

    normal_map_image = np.reshape(normal_map_image, (height, width, 3))
    normal_map_image = normalize_normal_map(normal_map_image)
    normal_map_image = normal_map_image * 0.5 + 0.5  # transforms from [-1,1] to [0,1]
    normal_map_image *= 255
    normal_map_image = normal_map_image.astype(np.uint8)

    return normal_map_image


def normalize_normal_map(normal_map: np.ndarray) -> np.ndarray:
    """Normalizes a normal mapping.

    Parameters
    ----------
    normal_map : np.ndarray
        The normal mapping.

    Returns
    -------
    np.ndarray
        The normalized normal mapping.
    """
    normal_map_length_map = np.sqrt(
        normal_map[:, :, 0] ** 2 + normal_map[:, :, 1] ** 2 + normal_map[:, :, 2] ** 2
    )

    for channel in range(normal_map.shape[2]):
        normal_map[:, :, channel][normal_map_length_map != 0] /= normal_map_length_map[
            normal_map_length_map != 0
        ]

    normal_map[normal_map_length_map == 0] = [0, 0, 1]

    return normal_map


def decode_normal_map_image(normal_map_image: np.ndarray) -> np.ndarray:
    """Decodes a normal map image.

    Parameters
    ----------
    normal_map_image : np.ndarray
        The normal map image to decode.

    Returns
    -------
    np.ndarray
        The decoded normal map image.
    """
    normal_map = normal_map_image.astype(np.float64) / 255
    normal_map = (normal_map - 0.5) * 2

    normal_map = normalize_normal_map(normal_map)

    return normal_map


def normal_map_high_pass(
    normal_map_image: np.ndarray, blur_kernel_radius: int
) -> np.ndarray:
    import cv2 as cv

    blur_kernel_size = (blur_kernel_radius * 2 + 1, blur_kernel_radius * 2 + 1)
    blurred_normal_map_image = cv.GaussianBlur(normal_map_image, blur_kernel_size, 0)

    normal_map = decode_normal_map_image(normal_map_image)
    blurred_normal_map = decode_normal_map_image(blurred_normal_map_image)
    blurred_normal_map = normalize_normal_map(blurred_normal_map)

    blurred_normal_map -= [0, 0, 1]
    high_frequency_normal_map = normal_map - blurred_normal_map
    high_frequency_normal_map = normalize_normal_map(high_frequency_normal_map)

    return ((high_frequency_normal_map * 0.5 + 0.5) * 255).astype(np.uint8)
