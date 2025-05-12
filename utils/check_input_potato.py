import numpy as np
import cv2
import os

verbose = False

def load_and_inspect_npy(file_path):
    # Load the .npy file
    data = np.load(file_path)

    if verbose:
        # Print the file path
        print(file_path)
        # Inspect the content
        # print("Data type:", type(data))
        print("Shape:", data.shape)
        # print("Content:", data)
        # Print min, max and std
        print("Min:", np.min(data))
        print("Max:", np.max(data))
        print("Std:", np.std(data))
        print("--------------------------------------------------")
        # Print the values greater than 1.0
        print("Values greater than 1.0:", data > 1.0)
        # print percentage of values greater than 1.0
        print("Percentage of values greater than 1.0:", np.sum(data > 1.0) / data.size)
        print("Percentage of values lower than 0.0:", np.sum(data < 0.0) / data.size)
        print("\n")

    return data

def compute_aolp(cos_aolp, sin_aolp):
    aolp = np.arctan2(sin_aolp, cos_aolp)
    print(f"AoLP min and max: {np.min(aolp)}, {np.max(aolp)}")
    # aolp = aolp + np.pi/2
    # aolp = np.clip(aolp, 0, np.pi)
    return aolp

def preprocess_dolp(dolp):
    # dolp = np.clip(dolp, 0, 1)
    return dolp

def create_dolp_image(dolp):
    """ normalize and use hot color map """
    # dolp = dolp / np.max(dolp)
    dolp = cv2.applyColorMap((dolp * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return dolp

def create_aolp_image(aolp):
    """ normalize and use hot color map """
    # Normalize aolp to [0, 1] for hue
    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    # Value is set to 1 (maximum brightness)
    saturation = (np.ones_like(hue)*255).astype(np.uint8)
    # Normalize dolp to [0, 1] for saturation
    value = np.ones_like(hue) * 255

    hsv_image = np.stack((hue, saturation, value), axis=-1)

    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return rgb_image


def create_pol_image(dolp, aolp):
    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    value = (np.clip(dolp, 0, 1)*255).astype(np.uint8)
    saturation = np.ones_like(hue) * 255

    hsv_image = np.stack((hue, saturation, value), axis=-1)

    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return rgb_image


def concat_and_save(img1, img2, img3, img4, output_path):
    # Concatenate the color and polarization images
    top = np.concatenate((img1, img2), axis=1)
    bootom = np.concatenate((img3, img4), axis=1)
    output_image = np.concatenate((top, bootom), axis=0)

    cv2.imwrite(output_path, output_image)

    return output_image


mcubes_folder = '../potato/'
count = 0
# Iterate through the files in the MCubeS folder
filenames = os.listdir(mcubes_folder + 'pol_dolp/')
sorted_filenames = sorted(filenames)
for filename in sorted_filenames:
    count += 1
    #if count > 100:
    #    break
    if filename.endswith('.npy'):
        print(filename)
        output_path = 'output/' + filename + '.png'

        dolp_file = mcubes_folder + 'pol_dolp/' + filename
        dolp = load_and_inspect_npy(dolp_file)
        dolp = preprocess_dolp(dolp)

        cos_file = mcubes_folder + 'pol_aolp_cos/' + filename
        cos_aolp = load_and_inspect_npy(cos_file)

        sin_file = mcubes_folder + 'pol_aolp_sin/' + filename
        sin_aolp = load_and_inspect_npy(sin_file)

        color_image = cv2.imread(mcubes_folder + 'pol_color/' + filename.replace('.npy', '.png'))

        aolp = compute_aolp(cos_aolp, sin_aolp)

        pol_image = create_pol_image(dolp, aolp)
        aolp_image = create_aolp_image(aolp)
        dolp_image = create_dolp_image(dolp)
        concat_and_save(color_image, pol_image, dolp_image, aolp_image, output_path)
        
