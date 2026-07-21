import cv2
import os

# =====================================================================
# 1. Charger une image ou capturer depuis la webcam
# =====================================================================
# CONFIGURATION : Mettez True pour utiliser la webcam, False pour charger une image
UTILISER_WEBCAM = False 

if UTILISER_WEBCAM:
    print("Initialisation de la webcam...")
    cap = cv2.VideoCapture(0)  # Ouvre la caméra par défaut
    
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        exit()
        
    print("Appuyez sur la touche 'ESPACE' pour capturer l'image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de la vidéo.")
            break
            
        cv2.imshow("Webcam - Appuyez sur Espace pour capturer", frame)
        
        # 32 correspond au code ASCII de la barre d'espace
        if cv2.waitKey(1) & 0xFF == 32:
            img = frame
            break
            
    cap.release()
    cv2.destroyAllWindows()
else:
    # Chargement d'une image depuis le disque (Ex: 'image.jpg')
    nom_image = 'image.jpg'
    img = cv2.imread(nom_image)

# Vérification de sécurité si l'image n'a pas pu être chargée
if img is None:
    print("Erreur : L'image source n'a pas pu être chargée.")
    exit()

# =====================================================================
# 2. Convertir en niveaux de gris
# =====================================================================
image_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =====================================================================
# 3. Appliquer un filtre (Gaussien ou Médian)[cite: 1]
# =====================================================================
# Option A : Filtre Gaussien (pour le bruit flou et lisser l'image)[cite: 1]
image_filtree = cv2.GaussianBlur(image_gris, (5, 5), 0)

# Option B : Si vous préférez le filtre Médian, décommentez la ligne suivante :
# image_filtree = cv2.medianBlur(image_gris, 5)

# =====================================================================
# 4. Redimensionner l'image en 320 x 240[cite: 1]
# =====================================================================
# Attention : OpenCV utilise le format (Largeur, Hauteur) pour cv2.resize
dimensions_cibles = (320, 240)
image_redimensionnee = cv2.resize(image_filtree, dimensions_cibles, interpolation=cv2.INTER_AREA)

# =====================================================================
# 5. Sauvegarder l'image prétraitée[cite: 1]
# =====================================================================
nom_sortie = 'image_pretraitee.jpg'
cv2.imwrite(nom_sortie, image_redimensionnee)
print(f"Opération réussie ! Image enregistrée sous : '{nom_sortie}'")

# =====================================================================
# Affichage des résultats (Visualisation)
# =====================================================================
cv2.imshow("1. Image Originale", img)
cv2.imshow("2 & 3. Gris et Filtre", image_filtree)
cv2.imshow("4. Redimensionnee (320x240)", image_redimensionnee)

print("\nCliquez sur l'une des fenêtres et appuyez sur n'importe quelle touche pour fermer.")
cv2.waitKey(0)
cv2.destroyAllWindows()