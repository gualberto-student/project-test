import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, roc_curve, auc

# ==========================================
# CONFIGURATION DES HYPERPARAMÈTRES
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Le transfert learning converge très rapidement (généralement 5 à 10 époques suffisent)

# ==========================================
# 1. TÉLÉCHARGEMENT ET NETTOYAGE STRICT (Votre logique)
# ==========================================
print("📥 Téléchargement direct du dataset Cats vs Dogs...")
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_dir = tf.keras.utils.get_file('cats_vs_dogs.zip', origin=url, extract=True)

base_extract_dir = os.path.dirname(zip_dir)

# Recherche automatique du vrai chemin contenant 'Cat' et 'Dog'
data_dir = None
for root, dirs, files in os.walk(base_extract_dir):
    if 'Cat' in dirs and 'Dog' in dirs:
        data_dir = root
        break

if data_dir is None:
    raise FileNotFoundError("Impossible de trouver les dossiers 'Cat' et 'Dog'.")

print(f"📂 Dataset trouvé avec succès dans : {data_dir}")

# Boucle de nettoyage des images corrompues
for folder in ['Cat', 'Dog']:
    folder_path = os.path.join(data_dir, folder)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            if os.path.isfile(fpath):
                if fname.startswith('.') or fname.endswith('.db'):
                    os.remove(fpath)
                    continue
                with open(fpath, 'rb') as fobj:
                    is_jfif = b"JFIF" in fobj.read(10)
                if not is_jfif:
                    os.remove(fpath)
        except Exception:
            try:
                os.remove(fpath)
            except:
                pass

# ==========================================
# 2. SÉPARATION EN TRAIN / VAL / TEST (80/10/10)
# ==========================================
print("📦 Répartition automatique : 80% Train, 10% Validation, 10% Test...")

# CORRECTION : Ajout de color_mode='rgb' pour forcer 3 canaux à la lecture
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=None,
    color_mode='rgb'  
)

rest_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=None,
    color_mode='rgb'  
)

total_rest = tf.data.experimental.cardinality(rest_ds).numpy()
val_size = total_rest // 2

val_ds = rest_ds.take(val_size)
test_ds = rest_ds.skip(val_size)

# Pipeline Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# CORRECTION : Fonction de sécurité pour garantir la forme shape [224, 224, 3]
def fix_channels_and_preprocess(image, label):
    image = tf.cast(image, tf.float32) # Convertit en float32 pour éviter d'autres erreurs de type
    image.set_shape([224, 224, 3])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label
def prepare_dataset(ds, augment=False, shuffle=False):
    # 1. On applique d'abord la correction de canal et le preprocess (sur des images individuelles)
    ds = ds.map(fix_channels_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. On mélange si besoin
    if shuffle:
        ds = ds.shuffle(1000)
        
    # 3. On fait le batch APRÈS
    ds = ds.batch(BATCH_SIZE)
    
    # 4. On applique l'augmentation sur le batch
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
train_dataset = prepare_dataset(train_ds, augment=True, shuffle=True)
val_dataset = prepare_dataset(val_ds)

# --- SÉCURISATION DU TEST SET VIA NUMPY ---
print("🔄 Extraction du jeu de test en tableaux NumPy...")

# On applique d'abord notre fonction de prétraitement sur les images individuelles du test set
test_dataset_preprocessed = test_ds.map(fix_channels_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

x_test_list, y_test_list = [], []
# On fait le batch APRÈS le prétraitement, ainsi tout est parfaitement aligné
for img, lbl in test_dataset_preprocessed.batch(BATCH_SIZE):
    x_test_list.append(img.numpy())
    y_test_list.append(lbl.numpy())

x_test = np.concatenate(x_test_list, axis=0)
y_true = np.concatenate(y_test_list, axis=0)
print(f"✅ Jeu de test converti avec succès ({x_test.shape[0]} images).")

# ==========================================
# 3. TRANSFER LEARNING (Exercice 2 - MobileNetV2)
# ==========================================
print("🏗️ Création du modèle basé sur MobileNetV2...")
# Chargement du modèle MobileNetV2 pré-entraîné sur ImageNet sans sa dernière couche
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False  # Bloquer les poids appris pour le transfert learning

# Construction de l'architecture finale
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 1 neurone de sortie avec activation Sigmoïde pour classification binaire
])

# Compilation du modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Fonction de perte binaire requise
    metrics=['accuracy']
)

# Callbacks optionnels (Early Stopping)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
]

# Entraînement
print("🚀 Début de l'entraînement...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 4. ÉVALUATIONS ET COURBE ROC (Demandé par le TP)
# ==========================================
print("📊 Évaluation finale et calcul des métriques...")

# Prédictions sur le jeu de test NumPy
y_pred_probs = model.predict(x_test).ravel()
y_pred_labels = (y_pred_probs > 0.5).astype(int)

# Rapport de classification (Précision, Rappel, F1-Score)
print("\n📋 Rapport de Classification :")
print(classification_report(y_true, y_pred_labels, target_names=['Cat', 'Dog']))

# Courbe ROC et calcul de l'AUC (Aire sous la courbe)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'MobileNetV2 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Classification Binaire')
plt.legend(loc='lower right')
plt.show()