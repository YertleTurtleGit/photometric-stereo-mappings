from typing import List, Tuple
import numpy as np


def light_table_opacity_alignment(
    light_table_opacity: np.ndarray,
    front_light_images: List[np.ndarray],
    back_light_images: List[np.ndarray],
) -> np.ndarray:
    import cv2 as cv

    def calculate_origin_opacity() -> np.ndarray:
        from DEFINITIONS import MINIMUM_COUNTRY_AREA, MINIMUM_COUNTRY_HOLE_AREA
        from photometric_stereo_mappings.mappings.mask import remove_small_islands

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

        threshold_light_all = cv.threshold(light_all, THRESHOLD, 255, cv.THRESH_BINARY)[
            1
        ]
        threshold_light_all = (threshold_light_all * 255).astype(np.uint8)
        upper_threshold_light_all = cv.threshold(
            light_all, UPPER_THRESHOLD, 255, cv.THRESH_BINARY
        )[1]
        upper_threshold_light_all = (upper_threshold_light_all * 255).astype(np.uint8)

        contours, hierarchy = cv.findContours(
            threshold_light_all, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        opacity_map = np.zeros(light_all.shape[:2], np.uint8)
        opacity_map_holes = np.zeros(light_all.shape[:2], np.uint8)

        for country in zip(contours, hierarchy[0]):
            contour = country[0]
            contour_hierarchy = country[1]

            has_parent: bool = contour_hierarchy[3] > -1

            if has_parent and cv.contourArea(contour) > MINIMUM_COUNTRY_HOLE_AREA:
                opacity_map_holes = cv.drawContours(
                    opacity_map_holes, [contour], 0, 255, cv.FILLED
                )

            if cv.contourArea(contour) > MINIMUM_COUNTRY_AREA:
                opacity_map = cv.drawContours(opacity_map, [contour], 0, 255, cv.FILLED)

        opacity_map[opacity_map_holes == 255] = 0
        # opacity_map = cv.morphologyEx(opacity_map, cv.MORPH_OPEN, np.ones((3, 3)))
        opacity_map = cv.medianBlur(opacity_map, 7)

        opacity_map = remove_small_islands(
            opacity_map,
            MINIMUM_COUNTRY_AREA,
            min_black_size=MINIMUM_COUNTRY_HOLE_AREA,
            black=True,
        )
        return opacity_map

    origin_opacity = calculate_origin_opacity()
    return bitmap_per_island_alignment(origin_opacity, light_table_opacity)


def bitmap_per_island_alignment(origin: np.ndarray, to_align: np.ndarray) -> np.ndarray:
    import cv2 as cv
    from utilities.image_file import rotate_contour

    to_align_contours, _ = cv.findContours(
        to_align,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    origin_contours, _ = cv.findContours(
        origin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    aligned = np.zeros((to_align.shape[0], to_align.shape[1], 3), np.uint8)
    # aligned = cv.drawContours(aligned, to_align_contours, -1, (255, 0, 0), cv.FILLED)

    for to_align_contour in to_align_contours:
        aligned = cv.drawContours(
            aligned,
            [np.int0(cv.boxPoints(cv.minAreaRect(to_align_contour)))],
            0,
            (255, 0, 0),
            7,
        )

        to_align_center, _, to_align_angle = cv.minAreaRect(to_align_contour)
        # to_align_angle = get_contour_angle(to_align_contour)

        lowest_distance = 500
        best_fit_origin_contour = None

        for origin_contour in origin_contours:
            origin_center, _, origin_angle = cv.minAreaRect(origin_contour)
            # origin_angle = get_contour_angle(origin_contour)

            origin_center = np.array(origin_center, np.float64)
            to_align_center = np.array(to_align_center, np.float64)

            distance = np.linalg.norm(origin_center - to_align_center)

            if distance < 500:  # TODO Find better value
                origin_area = cv.contourArea(origin_contour)
                to_align_area = cv.contourArea(to_align_contour)
                area_delta: float = abs(to_align_area - origin_area)

                if area_delta < origin_area * 0.5:  # TODO Find better value
                    if distance < lowest_distance:
                        lowest_distance = distance
                        best_fit_origin_contour = origin_contour
                        best_fit_origin_contour_angle = origin_angle
                        best_fit_origin_contour_center = origin_center

        if best_fit_origin_contour is not None:
            angle_difference: float = best_fit_origin_contour_angle - to_align_angle
            if angle_difference >= 80:
                angle_difference -= 90
            elif angle_difference <= -80:
                angle_difference += 90

            to_align_contour = np.array(
                rotate_contour(to_align_contour, angle_difference), np.float64
            )
            to_align_contour = np.array(to_align_contour, np.int32)

            """
            aligned = cv.drawContours(
                aligned,
                [np.int0(cv.boxPoints(cv.minAreaRect(best_fit_origin_contour)))],
                0,
                (0, 255, 0),
                3,
            )
            """
            aligned = cv.drawContours(
                aligned,
                [np.int0(cv.boxPoints(cv.minAreaRect(to_align_contour)))],
                0,
                (0, 0, 255),
                7,
                offset=(
                    round(best_fit_origin_contour_center[0] - to_align_center[0]),
                    round(best_fit_origin_contour_center[1] - to_align_center[1]),
                ),
            )

            # aligned = cv.drawContours(
            # aligned, [best_fit_origin_contour], 0, (0, 255, 0), cv.FILLED
            # )
            aligned = cv.drawContours(
                aligned,
                [to_align_contour],
                0,
                (255, 255, 255),
                cv.FILLED,
                offset=(
                    round(best_fit_origin_contour_center[0] - to_align_center[0]),
                    round(best_fit_origin_contour_center[1] - to_align_center[1]),
                ),
            )

    return aligned


def median_threshold_bitmap_alignment(images: List[np.ndarray]) -> List[np.ndarray]:
    """Aligns images using bit operations on median threshold bitmaps (MTBs).

    Parameters
    ----------
    images : List[np.ndarray]
        The array of images to be aligned.

    Returns
    -------
    List[np.ndarray]
        The array of aligned images.
    """
    from cv2 import createAlignMTB

    # if len(images[0].shape) > 2:
    #    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]

    alignMTB = createAlignMTB(cut=False, max_bits=1, exclude_range=1)
    # print(alignMTB.calculateShift(images[0], images[4]))
    alignMTB.process(images, images)

    return images


def aruco_homographic_alignment(
    images: List[np.ndarray],
    child_images: List[np.ndarray],
    number_of_used_markers: int = 8,
    debug_info: bool = False,
) -> List[np.ndarray]:
    """Aligns multiple images with aruco markers using homography.

    Parameters
    ----------
    images : List[np.ndarray]
        The input images to be aligned.
    child_images : List[np.ndarray]
        The child images to be aligned according to the images.
    marker_count : int, optional
        The number of markers to track. The default is 8.
        (This value is only to calculate the percentage of markers found.)
    debug_info : bool, optional
        If True, the function will print debug information. The default is False.

    Returns
    -------
    List[np.ndarray]
        The aligned output images.
    """
    from logging import warning
    from cv2 import aruco, findHomography, warpPerspective

    def get_aruco_corners(
        image: np.ndarray,
    ) -> List[Tuple[int, List[List[Tuple[float, float]]]]]:
        """Gets the aruco corners of an image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        List[Tuple[int, List[Tuple[float, float]]]]
            The aruco corners of the image with id and corner coordinates.
        """
        image = image[:, :, 1]

        all_corners: List[List[List[Tuple[float, float]]]] = []
        all_ids: List[int] = []

        for brightness_percent in range(10, 500, 10):
            brightness_image: np.ndarray = np.clip(
                image.astype(np.float64) * (brightness_percent / 100), 0, 255
            ).astype(np.uint8)

            # TODO Find good params: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
            aruco_params = aruco.DetectorParameters_create()
            aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            # aruco_params.useAruco3Detection = True
            aruco_dictionary = aruco.Dictionary_get(aruco.DICT_5X5_100)
            corners, ids, _ = aruco.detectMarkers(
                brightness_image, aruco_dictionary, parameters=aruco_params
            )

            if ids is not None:
                all_corners.extend(np.array(corners))
                all_ids.extend(ids.flatten())

        aruco_corners: List[Tuple[int, List[List[Tuple[float, float]]]]] = []
        existing_marker_ids: List[int] = []

        for i in range(len(all_ids)):
            marker_id: int = all_ids[i]
            marker_corners: List[List[Tuple[float, float]]] = all_corners[i]

            if marker_id not in existing_marker_ids:
                existing_marker_ids.append(marker_id)
                aruco_corners.append((marker_id, marker_corners))

        return aruco_corners

    images_marker_points: List[List[Tuple[int, List[List[Tuple[float, float]]]]]] = []
    feature_count_per_marker: List[int] = [0] * number_of_used_markers
    highest_marker_count: int = 0
    highest_marker_count_id: int = 0

    for i in range(len(images)):
        markers = get_aruco_corners(images[i])
        images_marker_points.append(markers)

        marker_count = len(markers)
        if marker_count > highest_marker_count:
            highest_marker_count = marker_count
            highest_marker_count_id = i

        for marker in markers:
            marker_id = marker[0]

            if marker_id > number_of_used_markers or marker_id <= 0:
                warning("Marker id is out of range: " + str(marker_id))
            else:
                feature_count_per_marker[marker_id - 1] += 1

    feature_count_per_image: List[int] = []

    for i in range(0, len(images_marker_points)):
        if i == highest_marker_count_id:
            continue

        reference_points: List[Tuple[float, float]] = []
        feature_points: List[Tuple[float, float]] = []

        for reference_candidate in images_marker_points[highest_marker_count_id]:
            for feature_candidate in images_marker_points[i]:
                if reference_candidate[0] == feature_candidate[0]:
                    for point_id in range(len(reference_candidate[1])):
                        reference_corners = reference_candidate[1][point_id]
                        feature_corners = feature_candidate[1][point_id]

                        reference_points.extend(reference_corners)
                        feature_points.extend(feature_corners)

        if len(feature_points) == 0:
            continue

        homography, _ = findHomography(
            np.array(feature_points), np.array(reference_points)
        )
        images[i] = warpPerspective(
            images[i], homography, (images[i].shape[1], images[i].shape[0])
        )
        if child_images is not None:
            child_images[i] = warpPerspective(
                child_images[i], homography, (images[i].shape[1], images[i].shape[0])
            )

        feature_count_per_image.append(len(feature_points))

    if debug_info:
        from matplotlib import pyplot as plt

        max_feature_points_per_image: int = number_of_used_markers * 4
        feature_count_per_image = (
            np.array(feature_count_per_image) / max_feature_points_per_image
        )
        feature_count_per_marker = np.array(feature_count_per_marker) / len(images)

        fig = plt.figure()
        fig.patch.set_facecolor((1, 1, 1))
        plt.title("relative feature count")
        plt.boxplot(
            [feature_count_per_image, feature_count_per_marker],
            labels=["per image", "per marker"],
            vert=False,
        )
        plt.xlabel(
            "per marker: "
            + str(feature_count_per_marker)
            + "\n"
            + "per image: "
            + str(feature_count_per_image)
        )
        plt.show()

    return images + child_images
