from typing import List, Tuple, Union
from pathlib import Path
import numpy as np


def read_image_dimensions(image_path: Path, raw: bool = False) -> Tuple[int, int]:
    if not image_path.is_file():
        raise FileNotFoundError(f"{image_path} does not exist.")

    if raw:
        from rawpy import imread as rawpy_imread

        rawpy_sizes = rawpy_imread(str(image_path)).sizes
        return rawpy_sizes.height, rawpy_sizes.width

        processed_image = rawpy_imread(str(image_path)).postprocess()
        return processed_image.shape[:2]
    else:
        from cv2 import imread

        return imread(str(image_path)).shape[:2]


def read_image(
    image_path: Path,
    raw: bool = False,
    raw_white_balance: Union[List[float], None] = None,
    original_bit_depth: bool = False,
    raw_ev: float = 0,
    scale: float = 1,
) -> np.ndarray:
    from cv2 import imread, cvtColor, COLOR_BGR2RGB, IMREAD_ANYCOLOR, IMREAD_ANYDEPTH

    if not image_path.is_file():
        raise FileNotFoundError(f"{image_path} does not exist.")

    if raw:
        import rawpy

        raw = rawpy.imread(str(image_path))
        black_level = raw.black_level_per_channel[0]

        image = raw.raw_image
        image = np.maximum(image, black_level) - black_level
        image *= 2**raw_ev
        image = image + black_level
        # image = np.minimum(image, 2**12 - 1)
        raw.raw_image[:, :] = image

        image = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=False if raw_white_balance else True,
            user_wb=raw_white_balance,
        )

        image = np.asarray(image, dtype=np.float64)
        image /= 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        return image

    if original_bit_depth:
        flags = IMREAD_ANYCOLOR | IMREAD_ANYDEPTH
    else:
        flags = IMREAD_ANYCOLOR

    image = imread(str(image_path), flags)
    if len(image.shape) == 3:
        image = cvtColor(image, COLOR_BGR2RGB)

    if scale != 1:
        from cv2 import resize

        image = resize(
            image, (round(image.shape[0] * scale), round(image.shape[1] * scale))
        )

    return image


def write_image(image: np.ndarray, image_path: Path) -> None:
    from cv2 import imwrite

    image_path.parent.mkdir(parents=True, exist_ok=True)

    if len(image.shape) == 3 and image.shape[2] == 3:
        from cv2 import cvtColor, COLOR_RGB2BGR

        image = cvtColor(image, COLOR_RGB2BGR)

    imwrite(str(image_path), image)


def show_image(image: np.ndarray, dpi: int = 100) -> None:
    from matplotlib import pyplot as plt

    cmap = "gray" if len(image.shape) == 2 else None

    fig = plt.figure()
    fig.set_dpi(dpi)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()


def show_images(
    images: List[np.ndarray], column_count: int = 4, dpi: int = 100
) -> None:
    from matplotlib import pyplot as plt
    from math import ceil

    if len(images) == 1:
        show_image(images[0])
    else:
        row_count = ceil(len(images) / column_count)
        fig = plt.figure(figsize=(column_count * 5, row_count * 3))
        fig.set_dpi(dpi)

        for i in range(len(images)):
            fig.add_subplot(row_count, column_count, i + 1)
            plt.imshow(images[i])
            plt.axis("off")

        plt.show()


def export_image_file(
    image_file_path: Path,
    target_file_path: Path,
    target_bit_depth: int,
    target_channel_count: int,
    target_image_width: int,
    gamma_correction: float = 1,
) -> None:
    image = read_image(image_file_path, original_bit_depth=True)

    if image.dtype == np.uint8 and target_bit_depth == 16:
        raise ValueError("Cannot perform a lossless conversion from 8 bit to 16 bit.")

    if image.dtype == np.uint16 and target_bit_depth == 8:
        image = image.astype(np.float64)
        image /= 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

    if target_channel_count == 1:
        if len(image.shape) == 3:
            image = image[:, :, 0]

    elif target_channel_count == 3:
        if len(image.shape) != 3 or image.shape[2] < 3:
            raise ValueError("Image must have a minimum of 3 channels.")
        image = image[:, :, :3]

    else:
        raise ValueError(f"Invalid channel count '{target_channel_count}'.")

    target_image_height = round(target_image_width * (image.shape[0] / image.shape[1]))
    target_resolution: Tuple[int, int] = (target_image_height, target_image_width)

    if image.shape[0] < target_resolution[0] or image.shape[1] < target_resolution[1]:
        raise ValueError(
            f"Image resolution is too small. ({target_resolution[1]}px is greater than {image.shape[1]}px)"
        )

    elif image.shape[0] > target_resolution[0] or image.shape[1] > target_resolution[1]:
        from cv2 import resize

        image = resize(image, (target_resolution[1], target_resolution[0]))

    if gamma_correction != 1:
        if target_bit_depth != 8:
            raise ValueError(
                "Cannot perform gamma correction on image with another target bit depth than 8 bit."
            )

        from cv2 import LUT

        table = np.array(
            [
                ((i / (2**8 - 1)) ** (1 / gamma_correction)) * (2**8 - 1)
                for i in np.arange(0, 2**8)
            ]
        ).astype("uint8")
        image = LUT(image, table)

    write_image(image, target_file_path)
