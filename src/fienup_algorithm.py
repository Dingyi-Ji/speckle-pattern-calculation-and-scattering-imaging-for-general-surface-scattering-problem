# Author: Ji Ding-yi
# Date: 2023-11-13

import matplotlib.pyplot as plt
import numpy as np
def fienup_phase_retrieval_3(speckle_pattern, mask=None, beta=0.8,
                             steps=3000, mode='hybrid', verbose=True, init_input=None, input_pattern=None,
                             speckle_mask=None, adaptive_beta=None):
    """
    Implementation of Fienup's phase-retrieval methods,version 3.
    Loop for every input phase pattern
    Parameters:
        mag: Measured magnitudes of Fourier transform
        mask: Binary array indicating where the image should be
              if padding is known
        beta: Positive step size
        steps: Number of iterations
        mode: Which algorithm to use
              (can be 'input-output', 'output-output' or 'hybrid')
        verbose: If True, progress is shown
        init_input: init input
        input_pattern: input pattern
        speckle_mask:

    Returns:
        x: Reconstructed image
        loss_list: Loss for every iteration


    """


    assert beta > 0, 'step size must be a positive number'
    assert steps > 0, 'steps must be a positive number'
    assert mode == 'input-output' or mode == 'output-output' \
           or mode == 'hybrid', \
        'mode must be \'input-output\', \'output-output\' or \'hybrid\''

    (_, _, speckle_pattern_number) = speckle_pattern.shape
    print(speckle_pattern_number)
    mag = speckle_pattern[:, :, 0]

    if mask is None:
        mask = np.ones(mag.shape)

    assert mag.shape == mask.shape, 'mask and mag must have same shape'

    # sample random phase and initialize image x
    y_hat = mag * np.exp(1j * 2 * np.pi * np.random.rand(*mag.shape))
    x = np.zeros(mag.shape)

    # previous iterate
    x_p = None

    if init_input is not None:
        # fourier transform
        x_hat = np.fft.fft2(init_input)

        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        y_hat = mag * np.exp(1j * np.angle(x_hat))

        x = init_input
        x_p = x

    if adaptive_beta is not None:
        beta_0 = beta

    # save loss value
    loss_list = np.zeros((steps * speckle_pattern_number,))
    # main loop
    loop_num = 0
    for i in range(1, steps + 1):
        # show progress
        if i % 10 == 0 and verbose:
            print("step", i, "of", steps)

        for j in range(1, speckle_pattern_number + 1):
            # print(j % speckle_pattern_number)
            # adaptive beta
            if adaptive_beta is not None:
                beta = (beta_0 - 0.3) * np.exp(-loop_num / steps) + 0.3

            mag = speckle_pattern[:, :, j % speckle_pattern_number]
            # inverse fourier transform
            y = np.fft.ifft2(y_hat) * np.matrix.conjugate(input_pattern[:, :, (j - 1) % speckle_pattern_number])

            # previous iterate
            if x_p is None:
                x_p = y
            else:
                x_p = x

                # updates for elements that satisfy object domain constraints
            if mode == "output-output" or mode == "hybrid":
                x = y

            # find elements that violate object domain constraints
            # or are not masked
            temp_logic = (np.abs(0.0018 - np.abs(y)) > 5e5)


            row, column = np.shape(temp_logic)
            indices = np.logical_or(np.logical_and(temp_logic, mask),
                                    np.logical_not(mask))

            # updates for elements that violate object domain constraints
            if mode == "hybrid" or mode == "input-output":
                x[indices] = x_p[indices] - beta * y[indices]
            elif mode == "output-output":
                x[indices] = y[indices] - beta * y[indices]

            # fourier transform

            temp_a = (input_pattern[:, :, j % speckle_pattern_number])
            plt.show()
            x_hat = np.fft.fft2(x * temp_a)

            loss_list[loop_num] = np.sum(np.abs(mag - np.abs(x_hat)))

            if speckle_mask is None:
                y_hat = mag * np.exp(1j * np.angle(x_hat))

            else:
                # In the case of insufficient speckle shooting range, the restoration effect can be improved by introducing correction factors
                # Ref:
                # Nishino, Yoshinori, Jianwei Miao, and Tetsuya Ishikawa.
                    # "Image reconstruction of nanostructured nonperiodic objects only from oversampled
                    # hard x-ray diffraction intensities." Physical Review B 68.22 (2003): 220101.

                pre_temp_mag = np.fft.fftshift(np.abs(x_hat))
                rectify_factor = speckle_mask * pre_temp_mag / (mag+1e-3)
                rectify_factor = np.mean(rectify_factor[rectify_factor != 0])
                rectify_factor = 0.7*(1/rectify_factor-1)+1
                temp_mag = np.fft.ifftshift(
                    speckle_mask * np.fft.fftshift(mag) + rectify_factor*(1 - speckle_mask)*pre_temp_mag)

                y_hat = temp_mag * np.exp(1j * np.angle(x_hat))

            loop_num = loop_num + 1

    return x, loss_list
