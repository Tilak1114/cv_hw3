import cv2
import color_conversion as clr_conv
import image_filtering as filter_hlpr

image = cv2.imread("Lenna.png")

vsh_img = clr_conv.bgr_to_vsh(image)

noisy_img = cv2.imread("Noisy_image.png", cv2.IMREAD_GRAYSCALE)
conv_image = filter_hlpr.apply_convolution(noisy_img)
avg_img = filter_hlpr.apply_average_filter(noisy_img)
gaussian_img = filter_hlpr.apply_gaussian_filter(noisy_img)
median_img = filter_hlpr.apply_median_filter(noisy_img)

cv2.imwrite("convolved_image.png", conv_image)
cv2.imwrite("average_image.png", avg_img)
cv2.imwrite("gaussian_image.png", gaussian_img)
cv2.imwrite("median_image.png", median_img)
