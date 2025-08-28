import numpy as np


def interpolate_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.interpolate import NearestNDInterpolator

    if len(image.shape) == 3:
        channels = [
            interpolate_image(image[:, :, channel_id], mask) for channel_id in range(3)
        ]
        return np.stack(channels, axis=2)

    interpolation_mask = np.where(mask > 0)

    interpolator = NearestNDInterpolator(
        np.transpose(interpolation_mask), image[interpolation_mask]
    )
    return interpolator(*np.indices(image.shape))


def edge_extend_image(image: np.ndarray, mask: np.ndarray, padding: int) -> np.ndarray:
    from cv2 import dilate

    dilate_kernel_size: int = padding * 2 * 2 + 1
    dilated_mask = dilate(mask, np.ones((dilate_kernel_size, dilate_kernel_size)))
    interpolation_mask = mask.copy()
    interpolation_mask[dilated_mask == 0] = 255
    interpolated_image = interpolate_image(image, interpolation_mask)
    interpolated_image[mask > 0] = image[mask > 0]
    return interpolated_image
