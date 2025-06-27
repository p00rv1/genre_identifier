from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Path to your JSON file
JSON_PATH = "data.json"

def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    mapping = data["mapping"]
    return X, y, mapping
def plot_pca(X, y, mapping):
    X_flat = X.reshape(X.shape[0], -1)  # Flatten MFCC
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=mapping)
    plt.title("PCA of MFCC Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
X, y, mapping = load_data(JSON_PATH)
plot_pca(X, y, mapping)
