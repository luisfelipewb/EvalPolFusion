import os
import cv2
import numpy as np

"""
When exporting masks from Label Studio, the names of the masks are changed to the labeling task names.
This script renames the masks in the 'masks' folder to the orignal names from 'all.txt'.
The key asssumption is that the names are sequential and match the order in 'all.txt'.
"""

# Path to the all.txt file and masks directory
all_txt_path = 'all.txt'
masks_dir = 'masks'
renamed_masks_dir = 'renamed_masks'

os.makedirs(renamed_masks_dir, exist_ok=True)

# Read the new names from all.txt
with open(all_txt_path, 'r') as f:
    new_names = [line.strip() for line in f if line.strip()]

# List all files in the masks directory
mask_files = sorted(os.listdir(masks_dir))

# Print each file in the masks folder
for mask_file, new_name in zip(mask_files, new_names):
    print(f"Renaming '{mask_file}' to '{new_name}.png'")

    src_path = os.path.join(masks_dir, mask_file)
    dst_path = os.path.join(renamed_masks_dir, f"{new_name}.png")
    # Open the image
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # Threshold the image: zeros stay zero, anything else becomes 255
    new_image = np.where(img == 0, 0, 255).astype(np.uint8)


    # Save the new image
    cv2.imwrite(dst_path, new_image)

