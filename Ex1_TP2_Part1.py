# =====================================================================
# 1. Importation des bibliothèques
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # Utilisé pour rendre la matrice de confusion plus lisible
# =====================================================================
# 2. Chargement et préparation des données
# =====================================================================
# A. Chargement du dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Définition des noms textuels des classes pour affichage
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# B. Normalisation des pixels (ramener les valeurs entre 0 et 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# C. Redimensionnement pour ajouter le canal unique (niveaux de gris)
X_train = np.expand_dims(X_train, -1)  # Donne une forme (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, -1)    # Donne une forme (10000, 28, 28, 1)

# D. Conversion des labels en One-Hot Encoding
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# E. Optionnel : Affichage de quelques exemples
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.suptitle("Exemples d'images du dataset Fashion-MNIST")
plt.show()

# =====================================================================
# 3. Construction du modèle CNN
# =====================================================================
model = models.Sequential()

# Première couche convolutionnelle (32 filtres, 3x3, ReLU)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Seconde couche convolutionnelle (64 filtres, 3x3, ReLU) + MaxPooling2D
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Couche Dropout pour éviter l'overfitting
model.add(layers.Dropout(0.25))

# Couche Flatten pour aplatir la matrice en vecteur
model.add(layers.Flatten())

# Couche Dense (128 neurones, ReLU)
model.add(layers.Dense(128, activation='relu'))

# Couche Dropout additionnelle pour la couche Dense (bonne pratique de régularisation)
model.add(layers.Dropout(0.5))

# Couche de sortie avec 10 neurones (softmax pour la classification multiclasse)
model.add(layers.Dense(10, activation='softmax'))

# Affichage du résumé de l'architecture
model.summary()

# =====================================================================
# 4. Compilation du modèle
# =====================================================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =====================================================================
# 5. Entraînement du modèle
# =====================================================================
# Entraînement sur 10 époques avec un batch size de 64
history = model.fit(X_train, y_train_encoded, 
                    epochs=10, 
                    batch_size=64, 
                    validation_data=(X_test, y_test_encoded))

# =====================================================================
# 6. Évaluation et analyse
# =====================================================================
# A. Évaluation sur les données de test
test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"\nPrécision finale sur le jeu de test : {test_acc * 100:.2f}%")

# B. Tracé des courbes d'accuracy et de loss
plt.figure(figsize=(12, 4))

# Courbe Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Courbe de Précision (Accuracy)')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()

# Courbe Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Courbe de Perte (Loss)')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()

plt.show()

# C. Matrice de confusion pour identifier les classes confondues
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion')
plt.ylabel('Vrais Labels')
plt.xlabel('Labels Prédits')
plt.show()

# Affichage du rapport de classification détaillé
print("\nRapport détaillé de Classification :")
print(classification_report(y_test, y_pred, target_names=class_names))