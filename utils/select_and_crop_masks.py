import numpy as np
import cv2
from pathlib import Path


def horizontal_crop(mask, y_start, y_end):
    """
    Crop the mask horizontally from y_start to y_end and from x_start to x_end.
    """
    # check mask dimentions
    return mask[y_start:y_end, :].copy()


def load_mask_binary(mask_path): #load the full mask, used for IoU
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"mask : {mask_path} not loaded")
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def count(mask):
    """
    Check if the mask is empty (all zeros).
    """
    if mask is None:
        return True
    return np.count_nonzero(mask)

empty_count = 0
first_band_count = 0
with_maks = 0

mask_sizes_list = []

# open the txt fille with the list of masks names in train set
train_set_path = '../potato/list_folder_old/train.txt'
test_set_path = '../potato/list_folder_old/test.txt'
val_set_path = '../potato/list_folder_old/val.txt'


all_masks_path = './all.txt'
all_masks_path = '../potato/list_folder_old/all.txt'


with open(train_set_path, 'r') as f:
    train_set = [line.strip() for line in f if line.strip()]

with open(test_set_path, 'r') as f:
    test_set = [line.strip() for line in f if line.strip()]

with open(val_set_path, 'r') as f:
    val_set = [line.strip() for line in f if line.strip()]

with open(all_masks_path, 'r') as f:
    all_set = [line.strip() for line in f if line.strip()]

selected_train_set = []
selected_test_set = []
selected_val_set = []

selected_mask_names = []

added_mask = None

if __name__=="__main__":

    masks_folder = '../potato/potato-seg/'
    masks = sorted(Path(masks_folder).glob("*.png"))


    # for mask in masks:
    for mask in all_set:
        mask_name = mask.replace("_mask.png", "")
        mask_path = f"{masks_folder}/{mask}.png"
        mask_gt = load_mask_binary(mask_path)

        # Area of interest
        start = 384
        height = 512
        
        cropped_mask = horizontal_crop(mask_gt, start, start + height)
   
        mask_count = count(cropped_mask)
        if mask_count == 0:
            empty_count += 1
            print(f"Mask: {mask_name} - Empty mask, count: {empty_count}")
            continue

        band_size = 20 # pixels in the top band to avoid tiny partial bottles
        first_band = count(cropped_mask[:band_size,:])
        if first_band > 0:
            first_band_count += 1
            print(f"Mask: {mask_name} - First band count: {first_band}")
            continue # skip if bottles in the top band ( partial bottles )
        
        with_maks += 1

        mask_size = np.count_nonzero(cropped_mask)
        if mask_size < 310:
            continue # skip small masks
        mask_sizes_list.append(mask_size)
        
        selected_mask_names.append(mask_name)
        if mask_name in train_set:
            selected_train_set.append(mask_name)
        elif mask_name in test_set:
            selected_test_set.append(mask_name)
        elif mask_name in val_set:
            selected_val_set.append(mask_name)
        else:
            print(f"Mask {mask_name} not found in any set")
            quit()
    
        cv2.imwrite(f"./newcrop/{mask_name}.png", cropped_mask)
        

print(f"Total masks processed: {len(masks)}")
print(f"Empty#: {empty_count}, FirstBand#: {first_band_count} Selected#: {with_maks}")
print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")
print(f"Train set size: {len(selected_train_set)}, Val set size: {len(selected_val_set)}, Test set size: {len(selected_test_set)}")
print(f"Len of selected masks {len(selected_mask_names)}")
print(f"Sum: {len(selected_train_set)+len(selected_val_set)+len(selected_test_set)}")


with open("selected_masks_new_crop.txt", "w") as f:
    for line in selected_mask_names:
        f.write(f"{line}\n")

mask_sizes_list.sort()
print(f"Smallest masks: {mask_sizes_list[:50]}")
