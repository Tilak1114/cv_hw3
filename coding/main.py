import cv2
import color_conversion as clr_conv
import image_filtering as filter_hlpr
import fourier_filtering as ff

image = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv_img = clr_conv.rgb_to_hsv(image)
hsv_img_2 = clr_conv.rgb_to_hsi(image)
cmyk_img = clr_conv.rgb_to_cmyk(image)
# vsh_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2RGB)
cv2.imwrite("hsv_image_1.png", hsv_img)
cv2.imwrite("hsv_image_2.png", hsv_img_2)
cv2.imwrite("cmyk_image.png", cmyk_img)

hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite("hsv_image_1_cv.png", hsvImg)

lab_img = clr_conv.rgb_to_lab(image)
cv2.imwrite("lab_image.png", lab_img)

noisy_img = cv2.imread("Noisy_image.png", cv2.IMREAD_GRAYSCALE)
conv_image = filter_hlpr.apply_convolution(noisy_img)
avg_img = filter_hlpr.apply_average_filter(noisy_img)
gaussian_img = filter_hlpr.apply_gaussian_filter(noisy_img)
median_img = filter_hlpr.apply_median_filter(noisy_img)

cv2.imwrite("convolved_image.png", conv_image)
cv2.imwrite("average_image.png", avg_img)
cv2.imwrite("gaussian_image.png", gaussian_img)
cv2.imwrite("median_image.png", median_img)

uexposed_img = cv2.imread("Uexposed.png", cv2.IMREAD_COLOR)
filter_hlpr.adjust_contrast_brightness(uexposed_img)
cv2.imwrite("adjusted_image.png", uexposed_img)

ft_img = ff.apply_fourier_transform(noisy_img)
cv2.imwrite("converted_fourier.png", ft_img)

