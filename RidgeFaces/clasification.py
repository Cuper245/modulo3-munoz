import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

# ==============================
# 1. CARGAR DATASET REAL
# ==============================
DATASET_DIR = "faces_32x32"   # cambia si tu carpeta tiene otro nombre

X, y = [], []

for label, folder in enumerate(sorted(os.listdir(DATASET_DIR))):
    folder_path = os.path.join(DATASET_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        X.append(img.flatten())  # 32x32 → 1024
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset cargado:")
print("X:", X.shape, "y:", y.shape)

n_classes = len(np.unique(y))

# ==============================
# 2. TRAIN / TEST + NORMALIZACIÓN
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# ==============================
# 3. SÍMPLEX REGULAR
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
# 4. RIDGE MANUAL
# ==============================
def ridge_regression(X, Y, alpha):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ Y)

# ==============================
# 5. BÚSQUEDA DE ALPHA
# ==============================
alphas = np.logspace(-3, 3, 50)
mse_list = []
betas = []

for a in alphas:
    beta = ridge_regression(X_train, Y_train, a)
    Y_pred = X_test @ beta

    mse = np.mean((Y_pred - T[y_test])**2)
    mse_list.append(mse)
    betas.append(beta)

best_alpha = alphas[np.argmin(mse_list)]
print("Mejor alpha:", best_alpha)

# ==============================
# 6. CLASIFICACIÓN
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
# 7. MSE vs ALPHA (OBLIGATORIO)
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
# 8. RIDGE PATH (OBLIGATORIO)
# ==============================
plt.figure()
for i in range(min(10, betas[0].shape[0])):  # primeros coeficientes
    coef_values = [b[i, 0] for b in betas]
    plt.semilogx(alphas, coef_values)

plt.axvline(best_alpha, linestyle="--")
plt.title("Ridge Path")
plt.xlabel("alpha")
plt.ylabel("Coeficientes")
plt.grid()
plt.show()

# ==============================
# 9. MATRIZ DE CONFUSIÓN (%)
# ==============================
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
plt.imshow(cm_norm, cmap="Blues")
plt.title("Matriz de Confusión (%)")
plt.xlabel("Predicho")
plt.ylabel("Real")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center")

plt.colorbar()
plt.show()

# ==============================
# 10. VISUALIZACIÓN RESULTADOS
# ==============================
idxs = np.random.choice(len(X_test), 12, replace=False)

fig, axes = plt.subplots(2,6, figsize=(12,5))

for ax, idx in zip(axes.flat, idxs):
    img = X_test[idx].reshape(32,32)
    real = y_test[idx]
    pred = y_pred[idx]

    color = "green" if real == pred else "red"

    ax.imshow(img, cmap="gray")
    ax.set_title(f"{real}->{pred}", color=color)
    ax.axis("off")

plt.suptitle("Predicciones (verde=correcto, rojo=error)")
plt.show()