import sys
import cv2
import argparse
import os
import numpy as np

image_folder = "../data/rgb/images/test"
mask_folder = "../data/masks/test"

def show_images(image_names, pred_folder):
    idx = 0
    n = len(image_names)
    while 0 <= idx < n:
        image_name = image_names[idx]
        image_path = os.path.join(image_folder, image_name + ".png")
        mask_path = os.path.join(mask_folder, image_name + "_mask.png")
        pred_path = os.path.join(pred_folder, image_name + ".png")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or find the image '{image_path}'")
            idx += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not open or find the mask '{mask_path}'")
            idx += 1
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print(f"Error: Could not open or find the prediction '{pred_path}'")
            idx += 1
            continue

        # Resize mask/pred to match image if needed
        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if pred.shape[0] != image.shape[0] or pred.shape[1] != image.shape[1]:
            pred = cv2.resize(pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Prepare left image: RGB with mask overlay (mask in blue)
        left_img = image.copy()
        mask_binary = (mask > 0).astype("uint8") * 255
        mask_colored = cv2.merge([mask_binary, np.zeros_like(mask_binary), np.zeros_like(mask_binary)])
        left_img = cv2.addWeighted(left_img, 1.0, mask_colored, 0.9, 0)

        # Prepare right image: mask (blue) + pred (red) on black
        right_img = np.zeros_like(image)
        right_img[:, :, 0] = mask_binary  # Blue 
        pred_binary = (pred > 0).astype("uint8") * 255
        right_img[:, :, 2] = pred_binary  # Red

        # Concatenate left and right images
        concat_img = cv2.hconcat([left_img, right_img])
        cv2.imshow('View Predictions', concat_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit early
            break
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            idx -= 1
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            idx += 1
        else:
            idx += 1  # Default: next image
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="View image, mask, and prediction overlays by image name.")
    parser.add_argument("image_name", nargs="?", default=None, help="Image name without extension (e.g., exp05_frame042323)")
    parser.add_argument("--pred_folder", default="../data/rgb/images/test",
                        help="Folder containing prediction images (default: ../data/rgb/images/test)")
    args = parser.parse_args()

    if args.image_name is not None:
        image_names = [args.image_name]
    else:
        # List all .png files in pred_folder, strip extension
        image_names = [os.path.splitext(f)[0] for f in os.listdir(args.pred_folder) if f.endswith(".png")]
        image_names.sort()

    if not image_names:
        print("No images found to display.")
        sys.exit(1)

    show_images(image_names, args.pred_folder)

if __name__ == "__main__":
    main()
