import numpy as np


def bgr_to_vsh(image: np.ndarray):
    result = np.zeros([image.shape[1], image.shape[0], image.shape[2]], dtype=np.uint8)
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            b, g, r = image[i][j]

            b /= 255.0
            g /= 255.0
            r /= 255.0

            c_min = min(b, g, r)
            c_max = max(b, g, r)
            diff = c_max - c_min

            h, s, v = 0, 0, 0

            if c_max == c_min:
                h = 0
            elif c_max == r:
                h = (60 * ((g - b) / diff) + 360) % 360
            elif c_max == g:
                h = (60 * ((b - r) / diff) + 120) % 360
            elif c_max == b:
                h = (60 * ((r - g) / diff) + 240) % 360

            if c_max == 0:
                s = 0
            else:
                s = (diff / c_max) * 100

            v = c_max * 100

            result[i][j] = np.array([v, s, h/2])

    return result


def vsh_to_cmyk(image: np.ndarray):
    pass
