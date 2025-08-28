from typing import List, Tuple
import numpy as np
from deprecated import deprecated

try:
    from DEFINITIONS import MINIMUM_COUNTRY_AREA, MINIMUM_COUNTRY_HOLE_AREA
except ModuleNotFoundError:
    from photometric_stereo_mappings.DEFINITIONS import (
        MINIMUM_COUNTRY_AREA,
        MINIMUM_COUNTRY_HOLE_AREA,
    )


def remove_small_islands(
    mask: np.ndarray,
    min_white_size: int,
    min_black_size: int = -1,
    white: bool = True,
    black: bool = False,
) -> np.ndarray:
    """Removes small objects from an image.

    Parameters
    ----------
    image : np.ndarray
        The image to remove the small objects from.
    min_size : int
        The minimum size of the objects to remove.
    white : bool
        Whether to remove white objects.
    black : bool
        Whether to remove black objects.

    Returns
    ----------
    np.ndarray
        The image with the small objects removed.

    """

    if min_black_size == -1:
        min_black_size = min_white_size

    if black:
        mask = 255 - remove_small_islands(255 - mask, min_black_size)

    if white:
        from cv2 import (
            findContours,
            drawContours,
            contourArea,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE,
            FILLED,
        )

        contours, _ = findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)

        for _, contour in enumerate(contours):
            if contourArea(contour) < min_white_size:
                drawContours(mask, [contour], 0, 0, FILLED)

    return mask


@deprecated
def preserving_erosion(image: np.ndarray) -> np.ndarray:
    """Preserving erosion of an image.

    Parameters
    ----------
    image : np.ndarray
        The image to apply the erosion to.

    Returns
    -------
    np.ndarray
        The eroded image.
    """
    import cv2 as cv
    from skimage import morphology

    image = cv.erode(image, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    # image = cv.medianBlur(image, 45)

    skeleton = morphology.skeletonize(image.astype(np.float64) / 255)
    skeleton = (skeleton * 255).astype(np.uint8)
    skeleton = cv.dilate(skeleton, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

    image = cv.erode(image, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    restored_countries = np.logical_and(
        np.where(skeleton == 255, True, False), np.where(image == 0, True, False)
    )
    restored_countries_color = np.zeros(restored_countries.shape, dtype=np.uint8)
    restored_countries_color[restored_countries] = 255

    contours, _ = cv.findContours(
        restored_countries_color, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    restored_countries_color *= 0
    for _, contour in enumerate(contours):
        if cv.contourArea(contour) > 10**2:
            cv.drawContours(restored_countries_color, [contour], 0, 255, cv.FILLED)

    restored_countries[restored_countries_color == 0] = False
    image[restored_countries] = 255
    return image


@deprecated
def get_atlas_countries(
    mask: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    import cv2 as cv

    mask = mask.astype(np.uint8)

    country_masks: List[np.ndarray] = []
    country_rectangles: List[Tuple[int, int, int, int]] = []

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        contour_mask = np.zeros(mask.shape).astype(np.uint8)
        contour_mask = cv.drawContours(contour_mask, [contour], 0, 255, cv.FILLED)
        country_masks.append(contour_mask)

        r_y, r_x, r_h, r_w = cv.boundingRect(contour)
        country_rectangles.append((r_x, r_y, r_w, r_h))

    return country_masks, country_rectangles


def mask_from_frosted_glass(
    back_light_images: List[np.ndarray],
    minimum_country_area: int = MINIMUM_COUNTRY_AREA,
    minimum_country_hole_area: int = MINIMUM_COUNTRY_HOLE_AREA,
) -> np.ndarray:
    """Generates an mask from frosted glass images.

    Parameters
    ----------
    back_light_images : List[np.ndarray]
        The frosted glass images.
    minimum_country_area : int, optional
        The minimum_country area, by default MINIMUM_COUNTRY_AREA
    minimum_country_hole_area : int, optional
        The minimum country hole area, by default MINIMUM_COUNTRY_HOLE_AREA

    Returns
    -------
    np.ndarray
        The mask.
    """
    import cv2 as cv

    light_image = np.zeros(back_light_images[0].shape[:2])

    for back_light_image in back_light_images:
        back_light_image = cv.cvtColor(back_light_image, cv.COLOR_RGB2GRAY).astype(
            np.float64
        )
        light_image += back_light_image / len(back_light_images)

    # threshold_light_table = cv.threshold(
    #    light_image.astype(np.uint8),
    #    round((1 / 3) * 255),  # TODO Find good value.
    #    255,
    #    cv.THRESH_BINARY,
    # )[1]
    light_image = np.clip((255 - light_image).astype(np.uint8), 0, 255)
    return light_image

    threshold_light_table = (255 - threshold_light_table).astype(np.uint8)

    contours, hierarchy = cv.findContours(
        threshold_light_table, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(back_light_images[0].shape[:2], np.uint8)
    mask_holes = np.zeros(back_light_images[0].shape[:2], np.uint8)

    for country in zip(contours, hierarchy[0]):
        contour = country[0]
        contour_hierarchy = country[1]

        has_parent: bool = contour_hierarchy[3] > -1

        if has_parent and cv.contourArea(contour) > minimum_country_hole_area:
            mask_holes = cv.drawContours(mask_holes, [contour], 0, 255, cv.FILLED)

        if cv.contourArea(contour) > minimum_country_area:
            mask = cv.drawContours(mask, [contour], 0, 255, cv.FILLED)

    mask[mask_holes == 255] = 0

    mask = remove_small_islands(
        mask,
        minimum_country_area,
        min_black_size=minimum_country_hole_area,
        black=True,
    )

    return mask


def mask_from_light_table(
    light_table_image: np.ndarray,
    minimum_country_area: int = MINIMUM_COUNTRY_AREA,
    minimum_country_hole_area: int = MINIMUM_COUNTRY_HOLE_AREA,
    light_table_threshold: float = 1 / 2,
) -> np.ndarray:
    """Generates an mask from a light table image.

    Parameters
    ----------
    light_table_image : np.ndarray
        The light table image.
    minimum_country_area : int, optional
        The minimum country area, by default MINIMUM_COUNTRY_AREA
    minimum_country_hole_area : int, optional
        The minimum country hole area, by default MINIMUM_COUNTRY_HOLE_AREA
    light_table_threshold : float, optional
        The light table threshold, by default 1/2

    Returns
    -------
    np.ndarray
        The mask.
    """
    import cv2 as cv

    light_table_image = cv.cvtColor(light_table_image, cv.COLOR_RGB2GRAY)

    threshold_light_table = cv.threshold(
        light_table_image,
        round(light_table_threshold * 255),
        255,
        cv.THRESH_BINARY,
    )[1]

    threshold_light_table = (255 - threshold_light_table).astype(np.uint8)

    contours, hierarchy = cv.findContours(
        threshold_light_table, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(light_table_image.shape[:2], np.uint8)
    mask_holes = np.zeros(light_table_image.shape[:2], np.uint8)

    for country in zip(contours, hierarchy[0]):
        contour = country[0]
        contour_hierarchy = country[1]

        has_parent: bool = contour_hierarchy[3] > -1

        if has_parent and cv.contourArea(contour) > minimum_country_hole_area:
            mask_holes = cv.drawContours(mask_holes, [contour], 0, 255, cv.FILLED)

        if cv.contourArea(contour) > minimum_country_area:
            mask = cv.drawContours(mask, [contour], 0, 255, cv.FILLED)

    mask[mask_holes == 255] = 0

    mask = remove_small_islands(
        mask,
        minimum_country_area,
        min_black_size=minimum_country_hole_area,
        black=True,
    )

    return mask


@deprecated
def mask_from_front_and_back_light(
    front_light_images: List[np.ndarray],
    back_light_images: List[np.ndarray],
    minimum_country_area: int = MINIMUM_COUNTRY_AREA,
    minimum_country_hole_area: int = MINIMUM_COUNTRY_HOLE_AREA,
) -> np.ndarray:
    """Generates an mask from a front and back light images.

    Parameters
    ----------
    front_light_images : List[np.ndarray]
        The front light images.
    back_light_images : List[np.ndarray]
        The back light images.
    minimum_country_area : int, optional
        Minimum country area, by default MINIMUM_COUNTRY_AREA
    minimum_country_hole_area : int, optional
        Minimum country hole area, by default MINIMUM_COUNTRY_HOLE_AREA

    Returns
    -------
    np.ndarray
        The mask.
    """
    import cv2 as cv

    light_images: List[np.ndarray] = front_light_images + back_light_images
    light_image_count: int = len(light_images)
    light_all: np.ndarray = np.zeros(light_images[0].shape[:2])

    THRESHOLD: float = 11
    UPPER_THRESHOLD: float = 255 - THRESHOLD * 10

    for i in range(light_image_count):
        light_image = light_images[i].astype(np.uint8)
        light_all += cv.cvtColor(light_image, cv.COLOR_RGB2GRAY).astype(np.uint8)

    del light_images

    light_all = np.clip(light_all / light_image_count, 0, 255).astype(np.uint8)

    threshold_light_all = cv.threshold(light_all, THRESHOLD, 255, cv.THRESH_BINARY)[1]
    threshold_light_all = (threshold_light_all * 255).astype(np.uint8)
    upper_threshold_light_all = cv.threshold(
        light_all, UPPER_THRESHOLD, 255, cv.THRESH_BINARY
    )[1]
    upper_threshold_light_all = (upper_threshold_light_all * 255).astype(np.uint8)

    # Redefine cavities of countries with the aid of an adaptive threshold.
    adaptive_threshold_light_all = cv.adaptiveThreshold(
        light_all, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 3
    )
    adaptive_threshold_light_all = (adaptive_threshold_light_all * 255).astype(np.uint8)
    adaptive_threshold_light_all[threshold_light_all == 0] = 0
    adaptive_threshold_light_all[upper_threshold_light_all == 255] = 255
    contours, _ = cv.findContours(
        adaptive_threshold_light_all, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    adaptive_threshold_light_all = np.zeros(light_all.shape[:2], np.uint8)
    adaptive_threshold_light_all = cv.drawContours(
        adaptive_threshold_light_all, contours, -1, 255, cv.FILLED
    )

    contours, hierarchy = cv.findContours(
        threshold_light_all, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(light_all.shape[:2], np.uint8)
    mask_holes = np.zeros(light_all.shape[:2], np.uint8)

    for country in zip(contours, hierarchy[0]):
        contour = country[0]
        contour_hierarchy = country[1]

        has_parent: bool = contour_hierarchy[3] > -1

        if has_parent and cv.contourArea(contour) > minimum_country_hole_area:
            mask_holes = cv.drawContours(mask_holes, [contour], 0, 255, cv.FILLED)

        if cv.contourArea(contour) > minimum_country_area:
            mask = cv.drawContours(mask, [contour], 0, 255, cv.FILLED)

    threshold_light_all = preserving_erosion(threshold_light_all)
    mask[mask_holes == 255] = 0
    mask[adaptive_threshold_light_all == 0] = 0

    mask = remove_small_islands(
        mask,
        minimum_country_area,
        min_black_size=minimum_country_hole_area,
        black=True,
    )

    return mask
