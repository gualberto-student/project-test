# =====================================================================
# 1. Importation des bibliothèques
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# =====================================================================
# 2. Chargement et préparation des données
# =====================================================================
# A. Chargement du dataset MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# B. Normalisation des pixels (entre 0 et 1)
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# C. Redimensionnement en (28, 28, 1)
X_train_full = np.expand_dims(X_train_full, -1)
X_test = np.expand_dims(X_test, -1)

# D. Conversion des labels en one-hot encoding
y_train_full_encoded = to_categorical(y_train_full, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# E. Séparation des données en train / validation (Extraction de 10% pour la validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full_encoded, test_size=0.1, random_state=42
)

print(f"Train set: {X_train.shape} | Validation set: {X_val.shape} | Test set: {X_test.shape}\n")

# =====================================================================
# 3. Data Augmentation
# =====================================================================
# Explication : L'augmentation de données génère artificiellement de nouvelles variations 
# d'images à chaque époque. Cela empêche le réseau d'apprendre par cœur les pixels exacts (overfitting) 
# et le force à se focaliser sur la topologie globale du chiffre, le rendant plus robuste.
datagen = ImageDataGenerator(
    rotation_range=12,          # Rotation légère (10-15°)
    width_shift_range=0.1,      # Translation horizontale (10%)
    height_shift_range=0.1,     # Translation verticale (10%)
    zoom_range=0.1              # Zoom léger (10%)
)
# Ajustement du générateur aux données d'entraînement
datagen.fit(X_train)

# =====================================================================
# 4. Définition du modèle CNN Profond
# =====================================================================
model = models.Sequential()

# --- Bloc 1 ---
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# --- Bloc 2 ---
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# --- Bloc 3 (Optionnel mais recommandé pour un réseau profond) ---
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# --- Classification Fully Connected ---
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# =====================================================================
# 5. Compilation du modèle
# =====================================================================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =====================================================================
# 6. Entraînement et validation
# =====================================================================
# Entraînement en utilisant le flux d'images augmentées (datagen.flow)
BATCH_SIZE = 64
EPOCHS = 10  # Ajustable entre 5 et 15

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // BATCH_SIZE
)

# Affichez les courbes loss et accuracy pour train/validation
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Courbe')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Courbe')
plt.legend()
plt.show()

# =====================================================================
# 7. Évaluation sur le jeu de test & Analyse des erreurs
# =====================================================================
test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"\nPrécision finale sur le jeu de test : {test_acc * 100:.2f}%")

# Identification et affichage de quelques images mal classées
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Trouver les indices où la prédiction diffère du vrai label
indices_erreurs = np.where(y_pred != y_test)[0]

print(f"Nombre total d'erreurs sur le jeu de test : {len(indices_erreurs)}")

# Afficher les 5 premières erreurs
plt.figure(figsize=(10, 2))
for i, idx in enumerate(indices_erreurs[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.title(f"Vrai:{y_test[idx]} | Prédit:{y_pred[idx]}", fontsize=9)
    plt.axis('off')
plt.suptitle("Exemples d'images mal classées")
plt.show()