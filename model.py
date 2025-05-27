import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    inputs = np.array(data["mfcc"])  # Corrected: np.array() instead of np.array[]
    targets = np.array(data["labels"])
    return inputs, targets


if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)

    # Train-test split
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3, random_state=42  # Added random_state for reproducibility
    )

    # Model architecture
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),  # Corrected: "activation" (not "activations")
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")  # Assuming 10 output classes
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # Correct loss for integer labels
        metrics=["accuracy"]
    )

    model.summary()

    # Corrected: "targets_test" (not "targets_tests")
    model.fit(
        inputs_train, targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=50,
        batch_size=32
    )
# After training, save the model like this:
model.save("genre_classifier.keras")  # New recommended format
