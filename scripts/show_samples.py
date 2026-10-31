import os

import matplotlib.pyplot as plt

from src.data import CLASS_NAMES, read_images, read_labels

os.makedirs("figures", exist_ok=True)

for name in ["mnist", "fashion"]:
    x = read_images(f"data/{name}/train-images-idx3-ubyte.gz")
    y = read_labels(f"data/{name}/train-labels-idx1-ubyte.gz")

    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i], cmap="gray")
        ax.set_title(CLASS_NAMES[name][y[i]])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"figures/samples_{name}.png", dpi=150)
    plt.close(fig)
    print("Saved", f"figures/samples_{name}.png")
