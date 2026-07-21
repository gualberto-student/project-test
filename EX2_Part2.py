import tensorflow as tf
import os
import zipfile
import numpy as np
from tensorflow.keras import layers, models
# Vous pouvez enlever "import tensorflow_datasets as tfds" s'il n'est plus utilisé

# Configuration des hyperparamètres
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ==========================================
# 1. TÉLÉCHARGEMENT ET PRÉPARATION SÉCURISÉE (SANS TFDS)
# ==========================================
print("📥 Téléchargement direct du dataset Cats vs Dogs...")
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_dir = tf.keras.utils.get_file('cats_vs_dogs.zip', origin=url, extract=True)

# Le répertoire où les images sont extraites
# Le répertoire parent où Keras a extrait l'archive
base_extract_dir = os.path.dirname(zip_dir)

# Recherche automatique du vrai chemin contenant 'Cat' et 'Dog'
data_dir = None
for root, dirs, files in os.walk(base_extract_dir):
    if 'Cat' in dirs and 'Dog' in dirs:
        data_dir = root
        break

if data_dir is None:
    raise FileNotFoundError("Impossible de trouver les dossiers 'Cat' et 'Dog' dans l'archive extraite.")

print(f"📂 Dataset trouvé avec succès dans : {data_dir}")

# Boucle de nettoyage adaptée
for folder in ['Cat', 'Dog']:
    folder_path = os.path.join(data_dir, folder)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            if os.path.isfile(fpath):
                # Supprime les fichiers système inutiles comme Thumbs.db
                if fname.startswith('.') or fname.endswith('.db'):
                    os.remove(fpath)
                    continue
        except Exception:
            pass  # Ignore l'image si elle est illisible ou verrouillée
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            if os.path.isfile(fpath):
                # Supprime les fichiers système inutiles comme Thumbs.db
                if fname.startswith('.') or fname.endswith('.db'):
                    os.remove(fpath)
                    continue
                # Vérifie si l'image possède un en-tête valide
                with open(fpath, 'rb') as fobj:
                    is_jfif = b"JFIF" in fobj.read(10)
                if not is_jfif:
                    os.remove(fpath)
        except Exception:
            os.remove(fpath)

print("📦 Répartition automatique : 80% Train, 10% Validation, 10% Test...")
# Création des datasets en spécifiant les splits manuellement pour respecter vos proportions (80/10/10)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # On prend 20% pour le reste (Validation + Test)
    subset="training",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=None # Pas de batch pour pouvoir appliquer le pipeline et le découpage après
)

rest_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=None
)

# On sépare équitablement les 20% restants en 10% Val et 10% Test
# On calcule la taille totale du sous-ensemble de validation
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

def prepare_dataset(ds, augment=False, shuffle=False):
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_dataset = prepare_dataset(train_ds, augment=True, shuffle=True)
val_dataset = prepare_dataset(val_ds)

# --- SÉCURISATION DU TEST SET VIA NUMPY ---
print("🔄 Extraction du jeu de test en tableaux NumPy...")
x_test_list, y_test_list = [], []
for img, lbl in test_ds.batch(BATCH_SIZE):
    x_test_list.append(img.numpy())
    y_test_list.append(lbl.numpy())

x_test = np.concatenate(x_test_list, axis=0)
y_true = np.concatenate(y_test_list, axis=0)
print(f"✅ Jeu de test converti avec succès ({x_test.shape[0]} images).")