import numpy as np
import cv2


def apply_fourier_transform(image: np.ndarray):
    if image.ndim > 2:
        raise Exception("Required a grayscale image with 2 dimensions!")

    dft = cv2.dft(
        np.float32(
            get_padded_img_for_fourier(image)
        ),
        flags=cv2.DFT_COMPLEX_OUTPUT
    )
    dft_shift = dft

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


def get_padded_img_for_fourier(image: np.ndarray):
    padded_image = image.copy()
    bot_padding = np.zeros([image.shape[0], 2 * image.shape[1]], dtype=np.uint8)
    right_padding = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)

    padded_image = np.hstack([padded_image, right_padding])
    padded_image = np.vstack([padded_image, bot_padding])

    return padded_image
