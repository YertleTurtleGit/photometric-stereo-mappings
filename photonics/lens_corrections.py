from logging import warning
import numpy as np
from cv2 import remap, INTER_LANCZOS4
from lensfunpy import Database, Modifier
from typing import List


def correct_vignette(
    raw_images: List[np.ndarray],
    cam_maker: str,
    cam_model: str,
    lens_maker: str,
    lens_model: str,
    focal_length: int,
    aperture: float,
    distance: float,
) -> List[np.ndarray]:
    """Corrects vignette in raw images.

    Parameters
    ----------
    raw_images : List[np.ndarray]
        The input RAW images.
    cam_maker : str, optional
        The camera maker, by default CAM_MAKER
    cam_model : str, optional
        The camera model, by default CAM_MODEL
    lens_maker : str, optional
        The lens maker, by default LENS_MAKER
    lens_model : str, optional
        The lens model, by default LENS_MODEL
    focal_length : int, optional
        The focal length, by default FOCAL_LENGTH
    aperture : float, optional
        The aperture, by default APERTURE
    distance : float, optional
        The distance to the object, by default DISTANCE_TO_OBJECT

    Returns
    -------
    List[np.ndarray]
        The corrected images.
    """

    db = Database()
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]

    height, width, _ = raw_images[0].shape

    mod = Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    for raw_image in raw_images:
        vignette_is_applied = mod.apply_color_modification(raw_image)

        if not vignette_is_applied:
            warning("Vignette correction matrix not found in database.")
            return raw_images

    return raw_images


def correct_transversal_chromatic_aberration(
    raw_images: List[np.ndarray],
    cam_maker: str,
    cam_model: str,
    lens_maker: str,
    lens_model: str,
    focal_length: int,
    aperture: float,
    distance: float,
) -> List[np.ndarray]:
    """Correct transversal chromatic aberration in raw images.

    Parameters
    ----------
    raw_images : List[np.ndarray]
        The input RAW images.
    cam_maker : str
        The camera maker.
    cam_model : str
        The camera model.
    lens_maker : str
        The lens maker.
    lens_model : str
        The lens model.
    focal_length : int
        The focal length.
    aperture : float
        The aperture.
    distance : float
        The distance to the object.

    Returns
    -------
    List[np.ndarray]
        The corrected images.
    """

    db = Database()
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]

    height, width, _ = raw_images[0].shape

    mod = Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    ca_coords = mod.apply_subpixel_distortion()

    if ca_coords is None:
        warning(
            "Transversal chromatic aberration (TCA) correction matrix not found in database."
        )
        return raw_images

    for raw_image in raw_images:
        raw_image[:, :, 0] = remap(
            raw_image[:, :, 0], ca_coords[..., 0], None, INTER_LANCZOS4
        )
        raw_image[:, :, 1] = remap(
            raw_image[:, :, 1].swapaxes(0, 1), ca_coords[..., 1], None, INTER_LANCZOS4
        )
        raw_image[:, :, 2] = remap(
            raw_image[:, :, 2], ca_coords[..., 2], None, INTER_LANCZOS4
        )

    return raw_images


def correct_lens_distortion(
    raw_images: List[np.ndarray],
    cam_maker: str,
    cam_model: str,
    lens_maker: str,
    lens_model: str,
    focal_length: int,
    aperture: float,
    distance: float,
) -> List[np.ndarray]:
    db = Database()
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]

    height, width, _ = raw_images[0].shape

    mod = Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    undistorted_coords = mod.apply_geometry_distortion()

    if undistorted_coords is None:
        warning("Lens distortion correction matrix not found in database.")
        return raw_images

    for index, raw_image in enumerate(raw_images):
        raw_images[index] = remap(raw_image, undistorted_coords, None, INTER_LANCZOS4)

    return raw_images
