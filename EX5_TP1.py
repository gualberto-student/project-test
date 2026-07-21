import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay 
# =====================================================================
# 2. Charger le dataset
# =====================================================================
data = load_breast_cancer()
X = data.data        # Caractéristiques numériques (features) 
y = data.target      # Labels : 0 (malin) ou 1 (bénin) 

print("--- Informations sur le Dataset ---")
print("Taille du dataset :", X.shape)
print("Classes disponibles :", np.unique(y))
print("Noms des classes :", data.target_names)
print("-----------------------------------\n")

# =====================================================================
# 3. Split train/test (Division des données) 
# =====================================================================
# On garde 20% des données pour le test et 80% pour l'entraînement 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 4. Standardisation (Mise à l'échelle) 
# =====================================================================
# Étape essentielle pour le K-NN et le SVM afin que toutes les features aient la même importance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# =====================================================================
# 5. Classificateur K-NN (K-Nearest Neighbors) 
# =====================================================================
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)                   # Entraînement 
y_pred_knn = knn.predict(X_test)                  # Prédiction 
print(f"Précision K-NN : {accuracy_score(y_test, y_pred_knn) * 100:.2f}%") 

# =====================================================================
# 6. Classificateur SVM (Support Vector Machine) 
# =====================================================================
svm = SVC(kernel='linear') 
svm.fit(X_train, y_train)                   # Entraînement 
y_pred_svm = svm.predict(X_test)                  # Prédiction 
print(f"Précision SVM  : {accuracy_score(y_test, y_pred_svm) * 100:.2f}%") 

# =====================================================================
# 7. Classificateur Softmax (Régression Logistique Multinomiale) 
# =====================================================================
softmax = LogisticRegression(max_iter=1000) 
softmax.fit(X_train, y_train)               # Entraînement 
y_pred_softmax = softmax.predict(X_test)          # Prédiction 
print(f"Précision Softmax : {accuracy_score(y_test, y_pred_softmax) * 100:.2f}%\n")

# =====================================================================
# 8. Affichage des Matrices de Confusion 
# =====================================================================
# Configuration de l'affichage Matplotlib (3 graphiques côte à côte)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Matrice de confusion K-NN 
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, ax=axes[0], cmap='Blues', display_labels=data.target_names) 
axes[0].set_title("Matrice de confusion K-NN") 

# Matrice de confusion SVM 
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, ax=axes[1], cmap='Greens', display_labels=data.target_names) 
axes[1].set_title("Matrice de confusion SVM") 

# Matrice de confusion Softmax 
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_softmax, ax=axes[2], cmap='Oranges', display_labels=data.target_names) 
axes[2].set_title("Matrice de confusion Softmax") 

plt.tight_layout()
plt.show()