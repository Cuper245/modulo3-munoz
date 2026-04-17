import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(42)

IMG_SIZE = 32

# ==============================
# 1. PREPROCESAMIENTO
# ==============================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.equalizeHist(img)
    return img

# ==============================
# 2. CARGAR DATASET COMPLETO
# ==============================
def cargar_dataset(path):
    X, y = [], []

    for label, folder in enumerate(sorted(os.listdir(path))):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = preprocess(img)

            X.append(img.flatten())
            y.append(label)

    return np.array(X), np.array(y)

# ==============================
# 3. CARGAR TODO Y HACER SPLIT
# ==============================
DATASET_DIR = "dataset"

X, y = cargar_dataset(DATASET_DIR)

print("Dataset completo:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

n_classes = len(np.unique(y_train))

# ==============================
# 4. NORMALIZACIÓN
# ==============================
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8

X_train = (X_train - mu) / sigma
X_test  = (X_test - mu) / sigma

# guardar imágenes para visualización
X_test_images = X_test.copy()

# ==============================
# 5. PCA
# ==============================
k = min(120, X_train.shape[0] - 1)

U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
W = Vt[:k]

X_train = X_train @ W.T
X_test  = X_test @ W.T

print("Dimensión después de PCA:", X_train.shape)

# ==============================
# 6. SÍMPLEX REGULAR
# ==============================
def simplex_regular(m):
    T = np.zeros((m, m - 1))
    T[0, 0] = 1.0
    for i in range(1, m):
        T[i, 0] = -1.0 / (m - 1)
    for k in range(1, m - 1):
        T[k, k] = np.sqrt(1 - np.sum(T[k, :k]**2))
        for i in range(k + 1, m):
            T[i, k] = -T[k, k] / (m - k - 1)
    return T

T = simplex_regular(n_classes)
Y_train = T[y_train]

# ==============================
# 7. RIDGE
# ==============================
def ridge_regression(X, Y, alpha):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ Y)

# ==============================
# 8. BÚSQUEDA DE ALPHA
# ==============================
alphas = np.logspace(-2, 4, 50)
mse_list = []
betas = []

for a in alphas:
    beta = ridge_regression(X_train, Y_train, a)
    Y_pred_tmp = X_test @ beta

    mse = np.mean((Y_pred_tmp - T[y_test])**2)
    mse_list.append(mse)
    betas.append(beta)

best_alpha = alphas[np.argmin(mse_list)]
print("Mejor alpha:", best_alpha)

# ==============================
# 9. CLASIFICACIÓN
# ==============================
def predict_class(Y_pred, T):
    dists = np.linalg.norm(Y_pred[:, None] - T[None, :], axis=2)
    return np.argmin(dists, axis=1)

beta = ridge_regression(X_train, Y_train, best_alpha)
Y_pred = X_test @ beta
y_pred = predict_class(Y_pred, T)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# ==============================
# 10. MSE vs ALPHA
# ==============================
plt.figure()
plt.semilogx(alphas, mse_list)
plt.axvline(best_alpha, linestyle="--")
plt.title("MSE vs Alpha")
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.grid()
plt.show()

# ==============================
# 11. RIDGE PATH
# ==============================
plt.figure()
for i in range(min(10, betas[0].shape[0])):
    coef_values = [b[i, 0] for b in betas]
    plt.semilogx(alphas, coef_values)

plt.axvline(best_alpha, linestyle="--")
plt.title("Ridge Path")
plt.xlabel("alpha")
plt.ylabel("Coeficientes")
plt.grid()
plt.show()

# ==============================
# 12. MATRIZ DE CONFUSIÓN
# ==============================
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
plt.imshow(cm_norm, cmap="Blues")
plt.title(f"Matriz de Confusión (acc={acc:.2f})")
plt.xlabel("Predicho")
plt.ylabel("Real")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center")

plt.colorbar()
plt.show()

# ==============================
# 13. VISUALIZACIÓN
# ==============================
idxs = np.random.choice(len(X_test_images), 12, replace=False)

fig, axes = plt.subplots(2,6, figsize=(12,5))

for ax, idx in zip(axes.flat, idxs):
    img = X_test_images[idx].reshape(IMG_SIZE, IMG_SIZE)
    real = y_test[idx]
    pred = y_pred[idx]

    color = "green" if real == pred else "red"

    ax.imshow(img, cmap="gray")
    ax.set_title(f"{real}->{pred}", color=color)
    ax.axis("off")

plt.suptitle("Predicciones (verde=correcto, rojo=error)")
plt.show()