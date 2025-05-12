import cv2
import sys

# Vérifie que l'utilisateur a bien fourni un chemin en argument
if len(sys.argv) < 2:
    print("Usage : python affiche_image.py chemin/vers/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Lecture de l'image
image = cv2.imread(image_path)

# Vérification de la lecture
if image is None:
    print(f"Erreur : impossible de lire l'image à {image_path}")
    sys.exit(1)

# Affichage de l'image
cv2.imshow("Image", image)

# Attente jusqu'à ce que la fenêtre soit fermée
while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Permet de quitter avec la touche 'q'
        break

cv2.destroyAllWindows()
