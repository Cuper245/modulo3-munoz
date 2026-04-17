import cv2
import os

# ==============================
# CONFIG
# ==============================
INPUT_DIR = "identidad8"      # carpeta con fotos originales
OUTPUT_DIR = "identidad8_filtro"    # carpeta de salida
SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar detector de caras
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# FUNCION DE RECORTE
# ==============================
def recortar_cara(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None  # no detectó cara

    # tomar la cara más grande
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

    # expandir un poco el bounding box (mejor centrado)
    pad = int(0.2 * w)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    face = img[y1:y2, x1:x2]

    return face

# ==============================
# PROCESAR TODAS LAS IMÁGENES
# ==============================
count_ok = 0
count_fail = 0

for filename in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, filename)

    img = cv2.imread(path)
    if img is None:
        continue

    face = recortar_cara(img)

    if face is None:
        print(f"No detectada: {filename}")
        count_fail += 1
        continue

    # convertir a gris
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # redimensionar a 32x32
    face_resized = cv2.resize(face_gray, (SIZE, SIZE))

    # normalizar a [0,1] (opcional)
    face_norm = face_resized / 255.0

    # guardar
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, face_resized)

    count_ok += 1

print(f"\n✔ Caras detectadas: {count_ok}")
print(f"✖ Fallos: {count_fail}")