import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Baca citra input
input_image = imageio.imread("C:/Users/amara/Documents/yuba/download.jpeg", mode='L')  # Ubah as_gray menjadi mode='L'

# Ekualisasi Histogram
def histogram_equalization(image):
    # Hitung histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Hitung distribusi kumulatif
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalisasi

    # Terapkan ekualisasi
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
    img_eq = cdf[image.astype('uint8')]

    return img_eq

# Aplikasi Gaussian Filter untuk Pengurangan Noise
blurred_image = gaussian_filter(input_image, sigma=1)

# Terapkan ekualisasi histogram
equalized_image = histogram_equalization(blurred_image)

# Tampilkan hasil
plt.figure(figsize=(10, 7))
plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Citra Asli')
plt.subplot(1, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Citra dengan Filter Gaussian')
plt.subplot(1, 3, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Citra dengan Ekualisasi Histogram')
plt.show()
