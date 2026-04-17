import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import deque

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── PARÁMETROS ──────────────────────────────────────────────
IMG_SIZE = (64, 64)
DATA_DIR = ""
THRESHOLD = 0.6

# ── PREPROCESAMIENTO ────────────────────────────────────────
def procesar_imagen(img_np):
    img_np = cv2.resize(img_np, IMG_SIZE)
    edges = cv2.Canny(img_np, 100, 200)
    return edges.flatten() / 255.0

# ── CARGAR DATASET ──────────────────────────────────────────
def cargar_dataset(split="train"):
    X, y = [], []
    clases = {"izquierda": 0, "derecha": 1}
    
    for clase, etiqueta in clases.items():
        carpeta = os.path.join(DATA_DIR, split, clase)
        
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                ruta = os.path.join(carpeta, archivo)
                img = Image.open(ruta).convert("L")
                vector = procesar_imagen(np.array(img))
                X.append(vector)
                y.append(etiqueta)
    
    return np.array(X), np.array(y)

# ── ENTRENAR MODELO ─────────────────────────────────────────
def entrenar_modelo(X, y):
    global pca, scaler, modelo
    
    pca = PCA(n_components=100)
    X_p = pca.fit_transform(X)

    scaler = StandardScaler()
    X_p = scaler.fit_transform(X_p)

    modelo = LogisticRegression(max_iter=2000, class_weight='balanced')
    modelo.fit(X_p, y)

    return modelo

# ── CARGA ───────────────────────────────────────────────────
print("Cargando datos...")
X_train, y_train = cargar_dataset("train")
X_test, y_test = cargar_dataset("test")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

modelo = entrenar_modelo(X_train, y_train)

# ── EVALUACIÓN ──────────────────────────────────────────────
X_test_p = scaler.transform(pca.transform(X_test))
y_prob = modelo.predict_proba(X_test_p)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

print("\n====== REGRESIÓN LOGÍSTICA ======")
print(classification_report(y_test, y_pred,
                            target_names=["izquierda", "derecha"]))

# matriz
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["izq","der"],
            yticklabels=["izq","der"])
plt.title("Matriz de Confusión")
plt.show()

# ── ROC ─────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.grid()
plt.show()

print(f"AUC: {roc_auc:.4f}")

# ── FUNCIÓN DE SUAVIZADO PRO ────────────────────────────────
def decision_suavizada(hist):
    conteo = sum(hist)
    if conteo >= 10:
        return 1
    elif conteo <= 5:
        return 0
    else:
        return hist[-1]

# ── CÁMARA ──────────────────────────────────────────────────
print("\n Cámara activada")
print("i=izq | d=der | r=reentrenar | q=salir")

cap = cv2.VideoCapture(0)
historial = deque(maxlen=15)

X_extra, y_extra = [], []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #  ROI (centro)
        h, w = gray.shape
        gray = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

        vector = procesar_imagen(gray)
        vector_t = scaler.transform(pca.transform([vector]))

        prob = modelo.predict_proba(vector_t)[0][1]

        #  FILTRO POR CONFIANZA
        if prob > 0.7:
            pred = 1
        elif prob < 0.3:
            pred = 0
        else:
            pred = historial[-1] if len(historial) else 0

        historial.append(pred)
        pred_final = decision_suavizada(historial)

        label = "DERECHA" if pred_final else "IZQUIERDA"

        cv2.putText(frame, f"{label} ({prob:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Flechas IA", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('i'):
            X_extra.append(vector)
            y_extra.append(0)
            print("➕ IZQUIERDA")

        elif key == ord('d'):
            X_extra.append(vector)
            y_extra.append(1)
            print("➕ DERECHA")

        elif key == ord('r'):
            if len(X_extra) > 0:
                print(" Reentrenando...")
                X_train = np.vstack([X_train, np.array(X_extra)])
                y_train = np.hstack([y_train, np.array(y_extra)])

                modelo = entrenar_modelo(X_train, y_train)
                X_extra.clear()
                y_extra.clear()
                print(" Modelo actualizado")

        elif key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Interrumpido")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(" Cámara cerrada")