import os
import cv2
import numpy as np


# Load the png masks from the directory
mask_dir = '../multimodal_dataset/SS/'
mask_files = os.listdir(mask_dir)
mask_files = [mask_dir + file for file in mask_files if file.endswith('.png')]

labels_seen = set()
# Print the values and shape of the images in a numpy array
for mask_file in mask_files:
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    # print(mask.shape, mask.dtype, mask.min(), mask.max())
    # print(mask)
    # Print min and max values of the mask
    # print(mask.min(), mask.max())
    # print(mask.max())
    # Print the unique values in the mask
    unique_values = np.unique(mask)
    # add the unique values to the set
    labels_seen.update(unique_values)
    # print(f"Unique values {unique_values}")
    # Print the percentage of each one in the image
    # for i in unique_values:
        # print(f"Label: {i} -> {np.sum(mask == i) / mask.size:.3f}%")

print(f"Labels seen: {labels_seen}")
print(f"Number of labels: {len(labels_seen)}")
quit()

# Create another directory to save the adjusted masks
adjusted_mask_dir = '../potato/masks/'
os.makedirs(adjusted_mask_dir, exist_ok=True)

# For every image, replace the 255 value with 1 and save the png grayscale image
for mask_file in mask_files:
    # Read the image
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    # Replace 255 with 1
    mask[mask == 255] = 1
    # Save the image
    cv2.imwrite(adjusted_mask_dir + os.path.basename(mask_file), mask)
    print(f"Saved {adjusted_mask_dir + os.path.basename(mask_file)}")


