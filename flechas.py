import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ── PARÁMETROS ──────────────────────────────────────────────
IMG_SIZE = (64, 64)   # todas las imágenes se redimensionan igual
DATA_DIR = "modulo3-munoz"     # carpeta raíz

# ── FUNCIÓN: cargar imágenes de una carpeta ──────────────────
def cargar_dataset(split="train"):
    X, y = [], []
    clases = {"izquierda": 0, "derecha": 1}
    
    for clase, etiqueta in clases.items():
        carpeta = os.path.join(DATA_DIR, split, clase)
        
        for archivo in os.listdir(carpeta):
            if not archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            ruta = os.path.join(carpeta, archivo)
            img = Image.open(ruta).convert("L")      # escala de grises
            img = img.resize(IMG_SIZE)               # mismo tamaño
            vector = np.array(img).flatten() / 255.0 # píxeles como vector 0-1
            
            X.append(vector)
            y.append(etiqueta)
    
    return np.array(X), np.array(y)

# ── CARGAR DATOS ─────────────────────────────────────────────
print("Cargando datos...")
X_train, y_train = cargar_dataset("/train")
X_test,  y_test  = cargar_dataset("/test")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── ENTRENAR MODELO ──────────────────────────────────────────
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# ── EVALUAR ──────────────────────────────────────────────────
y_pred = modelo.predict(X_test)

print("\n── Reporte de clasificación ──")
print(classification_report(y_test, y_pred,
                             target_names=["izquierda", "derecha"]))

print("── Matriz de confusión ──")
print(confusion_matrix(y_test, y_pred))