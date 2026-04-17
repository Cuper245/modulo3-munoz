import os
import csv
from datetime import datetime

CARPETA_DESCARTES = "descartes"
CSV_SALIDA = "imagenes_descartadas.csv"

archivos_validos = (".jpg", ".jpeg", ".png", ".bmp")

with open(CSV_SALIDA, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["nombre_archivo", "razon", "fecha_captura"])

    for filename in sorted(os.listdir(CARPETA_DESCARTES)):
        if not filename.lower().endswith(archivos_validos):
            continue

        writer.writerow([
            filename,
            "Descartada por gesto diferente",
            datetime.now().strftime("%Y-%m-%d")
        ])

print(f"CSV generado: {CSV_SALIDA}")