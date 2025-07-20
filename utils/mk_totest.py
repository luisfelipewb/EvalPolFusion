import os
import shutil

# Chemin vers le fichier texte contenant la liste des noms d'images (sans extension)
txt_file_path = 'potato/list_folder/test.txt'

# Chemin vers le répertoire source (dossier A) où se trouvent les fichiers images
source_dir = 'potato/polL_color'

# Chemin vers le répertoire de destination (dossier B) où tu veux déplacer les fichiers
destination_dir = 'potato/totest'

# Assure-toi que le répertoire de destination existe
os.makedirs(destination_dir, exist_ok=True)

# Lis le fichier texte pour obtenir les noms des images
with open(txt_file_path, 'r') as f:
    image_names = f.readlines()

# Enlève les espaces et les sauts de ligne de chaque nom d'image
image_names = [name.strip() for name in image_names]

# Déplace les images du répertoire source vers le répertoire de destination
for image_name in image_names:
    source_file = os.path.join(source_dir, image_name + '.png')
    destination_file = os.path.join(destination_dir, image_name + '.png')
    
    # Vérifie si le fichier existe dans le répertoire source
    if os.path.exists(source_file):
        # Déplace le fichier vers le répertoire de destination
        shutil.copy(source_file, destination_file)
        print(f"Image déplacée: {image_name}.png")
    else:
        print(f"Image {image_name}.png non trouvée dans le répertoire source.")
