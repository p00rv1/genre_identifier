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
def plot_mfcc(mfcc, label, mapping):
    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc.T, cmap='viridis')
    plt.title(f"MFCC - {mapping[label]}")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()

# Load and plot
X, y, mapping = load_data(JSON_PATH)
plot_mfcc(X[0], y[0], mapping)
