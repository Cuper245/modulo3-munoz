import os
import cv2
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
n_identities = 6  # cuántas carpetas mostrar
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

folders = [
    f for f in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, f))
][:n_identities]

for ax, folder in zip(axes.flat, folders):
    folder_path = os.path.join(DATASET_DIR, folder)
    archivos = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if len(archivos) == 0:
        ax.axis("off")
        continue

    img_path = os.path.join(folder_path, archivos[0])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    ax.imshow(img, cmap="gray")
    ax.set_title(folder)
    ax.axis("off")

plt.suptitle("Muestra representativa del dataset real")
plt.tight_layout()
plt.savefig("muestra_dataset.png", dpi=300, bbox_inches="tight")
plt.show()