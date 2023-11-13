# Author: Ji Ding-yi
# Date: 2023-11-13

# Using multiple captured speckle images,
# phase retrieval algorithm is performed to recover unknown scattering parameters.

import numpy as np
import matplotlib.pyplot as plt
import src.fienup_algorithm as fienup
import src.correlation_calculation as correlation_calculation
from PIL import Image
from matplotlib import ticker
import time
import matplotlib as mpl

formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')

'''Parameter settings and image loading'''

# Parameter settings
SLM_discrete_num = 376  # The discrete range of valid values for SLM

# Does diffraction propagation using the angular spectrum method apply to the incident light?
angular_simu = False

# The number of random phase patterns.
random_pattern_number = 15

# The reduction factor of a reduction optical system created using a 4f system
# is multiplied by the pixel size of the SLM. This factor is used to equivalently
# compute the effect of reducing the SLM through the reduction optical system
reduce_index = 75 / 125  # A reduction optical system created using a 4f system

lamda = 532.47 * 1e-9
f_focus = 7.5 * 1e-2
slm_pixel_size = 8 * 1e-6 * reduce_index
ccd_pixel_size = 5.6 * 1e-6
k = 2 * np.pi / lamda

ccd_row = 492
ccd_column = 656

# The overall size of the SLM
SLM_row = 1200
SLM_column = 1920
# The size of padding applied to the SLM
row_pad_unilateral = int((SLM_row - SLM_discrete_num) / 2)
column_pad_unilateral = int((SLM_column - SLM_discrete_num) / 2)
# Is it circular?
circle = True

# Pre-computed size of padding and various parameters
N_all = int((lamda * f_focus) / (slm_pixel_size * ccd_pixel_size))
if N_all % 2 != 0:
    N_all = N_all + 1
print("N_all = ", N_all)
pad_num = int((N_all - SLM_discrete_num) / 2)
half_N_all = int(N_all / 2)
half_ccd_row = int(ccd_row / 2)
half_ccd_column = int(ccd_column / 2)
speckle_x_pad = int((N_all - half_ccd_row * 2) / 2)
speckle_y_pad = int((N_all - half_ccd_column * 2) / 2)

'''Speckle acquisition and image cropping'''

# Matrix initialization
ARRAY_input_light_pad = np.zeros((N_all, N_all, random_pattern_number), dtype="complex")
ARRAY_pad_amplitude_of_rec_catched_speckle = np.zeros((N_all, N_all, random_pattern_number))
mask = np.zeros((N_all, N_all))
speckle_mask = np.zeros((N_all, N_all))
# Loop reading
for loop in range(random_pattern_number):
    # it is necessary to loop and capture multiple incident image samples and speckle patterns

    # Loading known incident light patterns
    import_input_pattern_str = "E:/input_light_phase/random_input_" + str(loop) + ".bmp"
    input_pattern = plt.imread(import_input_pattern_str)

    # Loading captured speckle patterns
    import_catched_speckle_img_str = "E:/catched_speckle/catched_speckle_" + str(loop) + ".bmp"
    catched_speckle_img = plt.imread(import_catched_speckle_img_str)

    # Intercept the effective range of SLM
    slm_valid_part = input_pattern[row_pad_unilateral:row_pad_unilateral + SLM_discrete_num
    , column_pad_unilateral:column_pad_unilateral + SLM_discrete_num]
    phase_map = slm_valid_part / 255 * 2 * np.pi
    input_light = np.exp(1j * phase_map)
    if circle:
        for i in range(SLM_discrete_num):
            for j in range(SLM_discrete_num):
                if (i - SLM_discrete_num / 2) ** 2 + (j - SLM_discrete_num / 2) ** 2 > (SLM_discrete_num / 2) ** 2:
                    input_light[i][j] = 0

    '''Speckle acquisition and image cropping'''
    # Pad the incident light and determine if the pad value is correct
    if pad_num > 0:
        input_light_pad = np.pad(input_light, ((pad_num, pad_num)
                                               , (pad_num, pad_num)), 'constant')  # padding
    else:
        input_light_pad = input_light[-pad_num:SLM_discrete_num + pad_num, -pad_num:SLM_discrete_num + pad_num]

    r, c = input_light_pad.shape
    ############storage################
    ARRAY_input_light_pad[:, :, loop] = input_light_pad

    # make mask
    if 0 == loop:
        mask = np.ceil(np.abs(input_light_pad))

    '''Perform some pre processing on the captured speckle images, 
    such as adjusting light intensity, removing stray parts, etc'''

    rectified_catched_speckle = catched_speckle_img

    # Image flipping up and down(through a 4f system and lens)
    rectified_catched_speckle = np.flipud(rectified_catched_speckle)
    # Image flipping left and right(through a 4f system and lens)
    # rectified_catched_speckle = np.fliplr(rectified_catched_speckle)

    amplitude_of_rec_catched_speckle = np.power(rectified_catched_speckle, 0.5)

    if speckle_x_pad > 0 and speckle_y_pad > 0:

        pad_amplitude_of_rec_catched_speckle = np.pad(amplitude_of_rec_catched_speckle, ((speckle_x_pad, speckle_x_pad)
                                                                                         , (
                                                                                             speckle_y_pad,
                                                                                             speckle_y_pad)),
                                                      'constant')  # padding
    elif speckle_x_pad > 0:
        # print(1)
        pad_amplitude_of_rec_catched_speckle = np.pad(amplitude_of_rec_catched_speckle, ((speckle_x_pad, speckle_x_pad)
                                                                                         , (0, 0)),
                                                      'constant')  # padding
        pad_amplitude_of_rec_catched_speckle = pad_amplitude_of_rec_catched_speckle[:,
                                               -speckle_y_pad:ccd_column + speckle_y_pad]
    elif speckle_y_pad > 0:
        pad_amplitude_of_rec_catched_speckle = np.pad(amplitude_of_rec_catched_speckle, ((0, 0)
                                                                                         , (
                                                                                             speckle_y_pad,
                                                                                             speckle_y_pad)),
                                                      'constant')  # padding
        pad_amplitude_of_rec_catched_speckle = pad_amplitude_of_rec_catched_speckle[
                                               -speckle_x_pad:ccd_row + speckle_x_pad, :]
    else:
        pad_amplitude_of_rec_catched_speckle = amplitude_of_rec_catched_speckle[
                                               -speckle_x_pad:ccd_row + speckle_x_pad,
                                               -speckle_y_pad:ccd_column + speckle_y_pad]

    pad_amplitude_of_rec_catched_speckle = np.fft.ifftshift(pad_amplitude_of_rec_catched_speckle)

    ##########storage##########
    ARRAY_pad_amplitude_of_rec_catched_speckle[:, :, loop] = pad_amplitude_of_rec_catched_speckle
    if 0 == loop:
        delta_x = 0
        delta_y = 0
        temp_speckle_mask = np.ones((ccd_row - 2 * delta_x, ccd_column - 2 * delta_y))

        if speckle_x_pad > 0 and speckle_y_pad > 0:

            speckle_mask = np.pad(temp_speckle_mask, ((speckle_x_pad + delta_x, speckle_x_pad + delta_x)
                                                      , (speckle_y_pad + delta_y, speckle_y_pad + delta_y)),
                                  'constant')  # padding
        elif speckle_x_pad > 0:
            speckle_mask = np.pad(temp_speckle_mask,
                                  ((speckle_x_pad, speckle_x_pad)
                                   , (0, 0)),
                                  'constant')  # padding
            speckle_mask = speckle_mask[:,
                           -speckle_y_pad:ccd_column + speckle_y_pad]
        elif speckle_y_pad > 0:
            speckle_mask = np.pad(temp_speckle_mask, ((0, 0)
                                                      , (speckle_y_pad,
                                                         speckle_y_pad)),
                                  'constant')  # padding
            speckle_mask = speckle_mask[
                           -speckle_x_pad:ccd_row + speckle_x_pad, :]
        else:
            speckle_mask = temp_speckle_mask[
                           -speckle_x_pad:ccd_row + speckle_x_pad,
                           -speckle_y_pad:ccd_column + speckle_y_pad]

        middle_temp = int(N_all / 2)

print("loop import succeed")

plt.figure(1)
plt.imshow(rectified_catched_speckle)
plt.figure(2)
plt.imshow(ARRAY_pad_amplitude_of_rec_catched_speckle[:, :, 0])
plt.figure(3)
plt.imshow(speckle_mask)
plt.figure(4)
plt.imshow(np.fft.fftshift(ARRAY_pad_amplitude_of_rec_catched_speckle[:, :, 0]) * speckle_mask)
plt.show()

'''Execute phase recovery algorithm to recover the phase of the captured speckle'''
T1 = time.time()
retrieved_unknown_pattern, loss_list = fienup.fienup_phase_retrieval_3(
    speckle_pattern=(ARRAY_pad_amplitude_of_rec_catched_speckle),
    mask=mask,
    beta=0.7,
    steps=15,
    mode='hybrid',
    input_pattern=ARRAY_input_light_pad,
    speckle_mask=speckle_mask,
    adaptive_beta=False)
T2 = time.time()
print('Program runtime:%.2s sec' % ((T2 - T1)))
np.save("retrieved_unknown_pattern.npy", retrieved_unknown_pattern)
np.save("loss_list.npy", loss_list)

# retrieved_unknown_pattern = np.load("retrieved_unknown_pattern.npy")
# loss_list = np.load("loss_list.npy")

print(ARRAY_pad_amplitude_of_rec_catched_speckle[:, :, 0].shape)

retrieved_angle = np.angle(retrieved_unknown_pattern)

'''Restoration effect display'''
temp_n = 0  # Show from sheet 0 onwards

rectified_catched_speckle = ARRAY_pad_amplitude_of_rec_catched_speckle[:, :, temp_n] ** 2
input_light_pad = ARRAY_input_light_pad[:, :, temp_n]

plt.figure(3)
plt.imshow(rectified_catched_speckle, cmap="gray")

valid_row_start = int((N_all - SLM_discrete_num) / 2)
valid_row_end = int((N_all + SLM_discrete_num) / 2)
valid_column_start = int((N_all - SLM_discrete_num) / 2)
valid_column_end = int((N_all + SLM_discrete_num) / 2)

plt.rcParams['font.size'] = 6
plt.figure(5, figsize=(2, 2), dpi=300)
temp = np.abs(retrieved_unknown_pattern[valid_row_start:valid_row_end, valid_column_start:valid_column_end])
plt.imshow(temp, cmap="rainbow")
plt.xticks([])
plt.yticks([])
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))

t_max = np.max(temp)
t_min = np.min(temp)
delta_colorbar = (t_max - t_min) / 4
cb1 = plt.colorbar(format=formatter, fraction=0.045, pad=0.05)
cb1.set_ticks([t_min, t_max - 3 * delta_colorbar, t_max - 2 * delta_colorbar, t_max - delta_colorbar, t_max])
plt.xticks([])
plt.yticks([])
cb1.update_ticks()
plt.clim(0, 1)

plt.savefig('experiment setup 2 abs.svg', bbox_inches='tight', pad_inches=0.05)

plt.figure(7, figsize=(2, 2), dpi=300)
temp = retrieved_angle[valid_row_start:valid_row_end, valid_column_start:valid_column_end]
plt.imshow(temp, cmap="rainbow")
plt.xticks([])
plt.yticks([])
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cb1 = plt.colorbar(format=formatter, fraction=0.045, pad=0.05)
tick_locator = ticker.MaxNLocator(nbins=5)
cb1.locator = tick_locator
cb1.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
cb1.update_ticks()
plt.clim(-np.pi, np.pi)

plt.savefig('experiment setup 2 angle.svg', bbox_inches='tight', pad_inches=0.01)

plt.figure(8, figsize=(2, 2), dpi=300)
retrieved_speckle = np.abs(np.fft.fftshift(np.fft.fft2(input_light_pad * retrieved_unknown_pattern) ** 2))
plt.imshow(retrieved_speckle, cmap="gray")
plt.xticks([])
plt.yticks([])
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
plt.savefig('experiment setup 2 retrieved speckle.svg', bbox_inches='tight', pad_inches=0.05)

plt.figure(9)
plt.plot(loss_list, '*')

temp_speckle_1 = speckle_mask * np.fft.fftshift(rectified_catched_speckle)
temp_speckle_1 = temp_speckle_1[speckle_x_pad:N_all - speckle_x_pad, speckle_y_pad:N_all - speckle_y_pad]
temp_speckle_2 = speckle_mask * retrieved_speckle
temp_speckle_2 = temp_speckle_2[speckle_x_pad:N_all - speckle_x_pad, speckle_y_pad:N_all - speckle_y_pad]

correlation_coefficient = correlation_calculation.speckle_correlation_calculation(temp_speckle_1
                                                                                  , temp_speckle_2)
autocorrelation_array = correlation_calculation.speckle_correlation_calculation(
    temp_speckle_1
    , temp_speckle_2,
    dimension=2, Fft=True)
print("1,1", np.max(autocorrelation_array))
print("1,2", correlation_coefficient)

plt.figure(11)
plt.imshow((autocorrelation_array))

plt.figure(13, figsize=(2, 2), dpi=300)
temp = np.fft.fftshift(ARRAY_pad_amplitude_of_rec_catched_speckle[:, :, temp_n]) ** 2
plt.imshow(temp, cmap="gray")
plt.xticks([])
plt.yticks([])
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
plt.savefig('experiment setup 2 true speckle.svg', bbox_inches='tight', pad_inches=0.05)

plt.show()

'''Save the results'''
print("save image?[y/n]")
command_key = input()
if command_key == "y":
    # Number of random phase patterns
    random_pattern_number = 1024
    for loop in range(random_pattern_number):
        # multiple incident patterns and speckle patterns need to be collected in a loop

        import_input_pattern_str = "E:/input_light_phase/random_input_" + str(
            loop) + ".bmp"
        input_pattern = plt.imread(import_input_pattern_str)
        '''Intercept of phase as true value'''
        # Intercept the effective range of SLM
        slm_valid_part = input_pattern[row_pad_unilateral:row_pad_unilateral + SLM_discrete_num
        , column_pad_unilateral:column_pad_unilateral + SLM_discrete_num]
        phase_map = slm_valid_part / 255 * 2 * np.pi
        input_light = np.exp(1j * phase_map)
        if circle:
            for i in range(SLM_discrete_num):
                for j in range(SLM_discrete_num):
                    if (i - SLM_discrete_num / 2) ** 2 + (j - SLM_discrete_num / 2) ** 2 > (SLM_discrete_num / 2) ** 2:
                        input_light[i][j] = 0

        '''Speckle acquisition and image cropping'''
        # Pad the incident light and determine if the pad value is correct
        if pad_num > 0:
            input_light_pad = np.pad(input_light, ((pad_num, pad_num)
                                                   , (pad_num, pad_num)), 'constant')  # padding
        else:
            input_light_pad = input_light[-pad_num:SLM_discrete_num + pad_num, -pad_num:SLM_discrete_num + pad_num]

        retrieved_speckle = np.abs(np.fft.fftshift(np.fft.fft2(input_light_pad * retrieved_unknown_pattern) ** 2))

        save_str = "E:/retrieved_speckle_single_pixel_1/retrieved_speckle_" + str(loop) + ".bmp"

        half_N_all = int(N_all / 2)
        half_ccd_row = int(ccd_row / 2)
        half_ccd_column = int(ccd_column / 2)
        # Intercept the corresponding CCD area
        ccd_show = retrieved_speckle[half_N_all - half_ccd_row:half_N_all + half_ccd_row,
                   half_N_all - half_ccd_column:half_N_all + half_ccd_column]
        im = Image.fromarray(ccd_show)
        im = im.convert("L")
        im.save(save_str)
