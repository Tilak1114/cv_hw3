import numpy as np


def rgb_to_hsv(image: np.ndarray):
    result = np.zeros([image.shape[1], image.shape[0], image.shape[2]], dtype=np.uint8)
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            r, g, b = image[i][j]

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
                h = (60 * (g - b)) / diff
            elif c_max == g:
                h = 120 + (60 * (b - r) / diff)
            elif c_max == b:
                h = 240 + (60 * (r - g) / diff)

            if h < 0:
                h += 360

            if c_max == 0:
                s = 0
            else:
                s = (diff / c_max)

            v = c_max

            result[i][j] = np.array([h / 2, s * 255, v * 255])

    return result


def rgb_to_hsi(image: np.ndarray):
    result = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         r, g, b = image[i][j]
    #
    #         i = (r + g + b) / 3
    #         s = (1 - ((3 / (r + g + b)) * min(r, g, b)))
    #
    #         num = 0.5 * ((r - g) + (r - b))
    #         denom = np.sqrt(pow((r - g), 2) + (r - b) * (g - b))
    #
    #         theta = np.arccos(num // denom)
    #
    #         if b <= g:
    #             h = theta
    #         else:
    #             h = 360 - theta
    #
    #         # result[i][j] = np.array([r, g, b])

    return image


def rgb_to_cmyk(image: np.ndarray):
    result = np.zeros(
        [image.shape[0], image.shape[1], 4],
        dtype=np.uint8
    )
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i][j]

            r /= 250
            g /= 250
            b /= 250

            c = 1 - r
            m = 1 - g
            y = 1 - b
            k = min(r, g, b)

            result[i][j] = np.array([c, m, y, k])

    return result


def rgb_to_lab(image: np.ndarray):
    result = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i][j]

            r /= 250
            g /= 250
            b /= 250

            lab_matrix = np.array([[0.4124530, 0.3575800, 0.1804230],
                                   [0.2126710, 0.7151600, 0.0721690],
                                   [0.0193340, 0.1191930, 0.950227]])

            rgb_matrix = np.array([[r], [g], [b]])

            x, y, z = np.reshape(np.matmul(lab_matrix, rgb_matrix), [3])

            x = x / 0.950456
            z = z / 1.088754

            l, a, b = 0, 0, 0

            if y > 0.008856:
                l = 116 * np.cbrt(y) - 16
            else:
                l = 903.3 * y

            def fun_t(value):
                if value > 0.008856:
                    return np.cbrt(value)
                else:
                    return 7.787 * value + (16 / 116)

            a = 500 * (fun_t(x) - fun_t(y)) + 128
            b = 200 * (fun_t(y) - fun_t(z)) + 128

            result[i][j] = np.array([(l * 255) / 100, a + 128, b + 128])
    return result
