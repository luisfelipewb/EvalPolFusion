import os

# Remplace ce chemin par le chemin de ton dossier
folder_path = 'potato/totest'

# Parcours des fichiers dans le dossier
for filename in os.listdir(folder_path):
    if 'seg' in filename:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(f"Suppression de : {file_path}")
            os.remove(file_path)
