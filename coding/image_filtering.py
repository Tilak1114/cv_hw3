import numpy as np


def apply_convolution(image: np.ndarray):
    if image.ndim > 2:
        raise Exception("Required a grayscale image with 2 dimensions!")

    conv_filter = get_convolution_filter()
    padded_img = get_padded_image(image, conv_filter)
    return apply_filter(padded_img, conv_filter)


def apply_average_filter(image: np.ndarray):
    if image.ndim > 2:
        raise Exception("Required a grayscale image with 2 dimensions!")

    avg_filter = get_average_filter()
    return apply_filter(get_padded_image(image, avg_filter), avg_filter)


def apply_gaussian_filter(image: np.ndarray):
    if image.ndim > 2:
        raise Exception("Required a grayscale image with 2 dimensions!")

    gaussian_filter = get_gaussian_filter()
    return apply_filter(get_padded_image(image, gaussian_filter), gaussian_filter)


def apply_median_filter(image: np.ndarray):
    fltr = get_median_filter()
    padded_img = get_padded_image(image, fltr)
    result = np.zeros([padded_img.shape[0], padded_img.shape[1]], dtype=np.uint8)
    for i in range(padded_img.shape[0] - fltr.shape[0]):
        for j in range(padded_img.shape[1] - fltr.shape[1]):
            sample = padded_img[i:i + fltr.shape[1], j:j + fltr.shape[0]]
            median = np.median(sample)
            result[i + (fltr.shape[0] // 2)][j + (fltr.shape[1] // 2)] = median

    return result


def apply_filter(padded_img: np.ndarray, fltr: np.ndarray):
    result = np.zeros([padded_img.shape[0], padded_img.shape[1]], dtype=np.uint8)
    for i in range(padded_img.shape[0] - fltr.shape[0]):
        for j in range(padded_img.shape[1] - fltr.shape[1]):
            sample = padded_img[i:i + fltr.shape[0], j:j + fltr.shape[1]]
            conv_res_mat = np.multiply(sample, fltr)
            result[i + 1][j + 1] = np.sum(conv_res_mat)

    return result


def get_convolution_filter():
    return np.array([[1 / 9] * 3] * 3)


def get_average_filter():
    return np.array([[1 / 9] * 3] * 3)


def get_gaussian_filter():
    return (1 / 16) * np.array([[1, 2, 1], [1, 2, 1], [2, 4, 2]])


def get_median_filter():
    '''
    :return: 5X5 median filter that doesn't aid in
    convolution but is necessary to apply padding.
    '''
    return np.array([[1] * 5] * 5)


def get_padded_image(image: np.ndarray, fltr: np.ndarray):
    padded_img = image.copy()

    padding_size = fltr.shape[0] - 1

    for i in range(round(padding_size / 2)):
        padded_img = np.hstack(
            [
                np.hstack(
                    [
                        np.zeros([padded_img.shape[0], 1]),
                        padded_img
                    ]
                ),
                np.zeros(
                    [
                        padded_img.shape[0],
                        1
                    ]
                )]
        )
        padded_img = np.vstack(
            [
                np.vstack(
                    [
                        np.zeros(
                            [1, padded_img.shape[1]]
                        ), padded_img]
                ),
                np.zeros([1, padded_img.shape[1]])
            ]
        )

    return padded_img


def adjust_contrast_brightness(image: np.ndarray):
    brightness_factor = 25
    contrast_factor = 3

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i][j]

            b, g, r = (clip(contrast_factor * b + brightness_factor),
                       clip(contrast_factor * g + brightness_factor),
                       clip(contrast_factor * r + brightness_factor))

            image[i, j] = [b, g, r]

    return image


def clip(value):
    if value > 255:
        return 255
    if value < 0:
        return 0
    return value
