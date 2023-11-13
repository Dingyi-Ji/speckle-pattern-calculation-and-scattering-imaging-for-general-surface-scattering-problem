# Author: Ji Ding-yi
# Date: 2023-11-13

import numpy as np

def speckle_correlation_calculation(speckle_img_1, speckle_img_2, dimension=-1, Fft=False):
    # Calculate the Spearman correlation coefficient between the two
    # Calculate according to Freund's 1990 formula
    img_1_mean = np.mean(speckle_img_1)
    img_2_mean = np.mean(speckle_img_2)
    denominator = np.power(np.sum((speckle_img_1 - img_1_mean) ** 2)
                           * np.sum((speckle_img_2 - img_2_mean) ** 2), 0.5)
    if 2 == dimension:
        # The autocorrelation of two-dimensional images using cyclic padding method
        row, column = speckle_img_1.shape
        flip_array = np.flip(np.flip(np.conj(speckle_img_2-img_2_mean), axis=1), axis=0)
        if True == Fft:
            # Using fft method for larger images
            fft_mutiply = np.fft.fft2(speckle_img_1-img_1_mean) * np.fft.fft2(flip_array)
            autocorrelation_array = np.abs(np.fft.ifft2(fft_mutiply) )

            # Here we need to perform a flip, and the preliminary analysis is that the calculated value here is
            # actually R (- Mx, - Mi), and after the flip, it is R (Mx, My)
            autocorrelation_array = np.flip(np.flip(autocorrelation_array, axis=1), axis=0)
            print(denominator)
            return np.fft.fftshift(autocorrelation_array/denominator)

    numerator = np.sum((speckle_img_1 - img_1_mean) * (speckle_img_2 - img_2_mean))

    correlation_coefficient = numerator / denominator



    return correlation_coefficient


if __name__ == "__main__":
    img_1 = np.random.rand(10)
    img_2 = img_1 * 200 / 41 + 10
    correlation_coefficient = speckle_correlation_calculation(img_1, img_2)
    print(correlation_coefficient)
