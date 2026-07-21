import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor

# =====================================================================
# 1. Charger une image couleur & 2. Convertir en niveaux de gris
# =====================================================================
img_couleur = cv2.imread('image.jpg')

if img_couleur is None:
    print("Erreur : Impossible de charger 'image.jpg'.")
    exit()

img_gris = cv2.cvtColor(img_couleur, cv2.COLOR_BGR2GRAY)

# =====================================================================
# 3. Détecter les contours
# =====================================================================
# Utilisation du filtre de Canny (avec seuils adaptatifs 100 et 200)
contours = cv2.Canny(img_gris, 100, 200)

# =====================================================================
# 4. Calculer l'histogramme
# =====================================================================
# Calcul de l'histogramme avec OpenCV sur 256 niveaux [0-256]
hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256])

# =====================================================================
# 5. Détecter des points clés locaux (ORB)
# =====================================================================
# Initialisation du détecteur/descripteur ORB
orb = cv2.ORB_create()

# Détection des points clés et calcul des descripteurs
points_cles, descripteurs = orb.detectAndCompute(img_gris, None)

# Dessiner les points clés sur l'image d'origine pour la visualisation
img_orb = cv2.drawKeypoints(img_couleur, points_cles, None, color=(0, 255, 0), flags=0)

# =====================================================================
# 6. Appliquer la transformation de Fourier et le filtrage Gabor
# =====================================================================

# --- Transformation de Fourier Discrète (DFT) ---
# Passage au domaine fréquentiel
dft = cv2.dft(np.float32(img_gris), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Centrer les basses fréquences

# Calcul du spectre de magnitude pour affichage
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

# --- Filtrage de Gabor ---
# Application du filtre (fréquence spatiale = 0.6, orientation theta = 45 degrés soit pi/4)
gabor_reel, gabor_imag = gabor(img_gris, frequency=0.6, theta=np.pi/4)

# =====================================================================
# 7. Sauvegarder les résultats et Affichage
# =====================================================================
# Sauvegarde des fichiers images sur le disque
cv2.imwrite('resultat_contours.jpg', contours)
cv2.imwrite('resultat_orb.jpg', img_orb)

# Conversion du spectre de Fourier et Gabor en format 8 bits pour la sauvegarde
fourier_sauvegarde = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
gabor_sauvegarde = cv2.normalize(gabor_reel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite('resultat_fourier.jpg', fourier_sauvegarde)
cv2.imwrite('resultat_gabor.jpg', gabor_sauvegarde)

print("Tous les résultats ont été sauvegardés avec succès !")

# Affichage complet avec Matplotlib
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(img_couleur, cv2.COLOR_BGR2RGB)), plt.title('1. Image Couleur')
plt.subplot(2, 3, 2), plt.imshow(img_gris, cmap='gray'), plt.title('2. Niveaux de Gris')
plt.subplot(2, 3, 3), plt.imshow(contours, cmap='gray'), plt.title('3. Contours (Canny)')

plt.subplot(2, 3, 4)
plt.plot(hist, color='black'), plt.xlim([0, 256]), plt.title('4. Histogramme')

plt.subplot(2, 3, 5), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('6. Spectre Fourier')
plt.subplot(2, 3, 6), plt.imshow(gabor_reel, cmap='gray'), plt.title('6. Filtre Gabor (45°)')

plt.tight_layout()
plt.show()

# Fenêtre séparée pour afficher les points clés ORB
cv2.imshow("5. Points cles ORB", img_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()