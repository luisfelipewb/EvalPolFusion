import cv2
import numpy as np
import os
import sys

def remap_pixels(image_path, old_gray, new_gray):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire en niveaux de gris
    print("Type image d'entrée:", image.shape)
    
    if image is None:
        print(f"Erreur de lecture : {image_path}")
        return None

    # Remap des pixels (si nécessaire)
    image[image == old_gray] = new_gray

    print("Type image de sortie:", image.shape)
    return image

def batch_process(input_dir, output_dir, old_gray, new_gray):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            result = remap_pixels(input_path, old_gray, new_gray)
            if result is not None:
                cv2.imwrite(output_path, result)
                print(f"✔ {filename} → traité")
            else:
                print(f"✘ {filename} → ignoré")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("lensysargv",len(sys.argv) )
        print("Usage : python batch_remap_pixels.py dossier_entree ancienne_nuance nouvelle_nuance dossier_sortie")
        sys.exit(1)

    input_dir = sys.argv[1]
    old_gray = int(sys.argv[2])  # Ancienne nuance de gris
    new_gray = int(sys.argv[3])  # Nouvelle nuance de gris
    output_dir = sys.argv[4]

    batch_process(input_dir, output_dir, old_gray, new_gray)
