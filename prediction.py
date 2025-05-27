import librosa
import numpy as np
import tensorflow as tf
import json
import math

# Constants (should match your training parameters)
SAMPLE_RATE = 22050
DURATION = 30  # seconds


def load_model_and_mapping(model_path="genre_classifier.keras", json_path="data.json"):
    """Load the trained model and genre mapping"""
    try:
        model = tf.keras.models.load_model(model_path)
        with open(json_path, "r") as f:
            data = json.load(f)
            genre_mapping = data["mapping"]
        return model, genre_mapping
    except Exception as e:
        print(f"Error loading model or mapping: {e}")
        raise


def predict_genre(model, file_path, num_segments=10, n_mfcc=13, n_fft=2048, hop_length=512):
    """Predict genre from audio file"""
    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Calculate expected samples per segment
        samples_per_track = SAMPLE_RATE * DURATION
        num_samples_per_segment = int(samples_per_track / num_segments)
        expected_num_mfcc_vectors = math.ceil(num_samples_per_segment / hop_length)

        mfccs = []

        # Process segments
        for s in range(num_segments):
            start_sample = num_samples_per_segment * s
            finish_sample = start_sample + num_samples_per_segment

            if finish_sample > len(signal):
                continue

            segment = signal[start_sample:finish_sample]

            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=sr,
                n_fft=n_fft,
                n_mfcc=n_mfcc,
                hop_length=hop_length
            )
            mfcc = mfcc.T

            if len(mfcc) == expected_num_mfcc_vectors:
                mfccs.append(mfcc.tolist())

        if not mfccs:
            raise ValueError("No valid segments found in audio file")

        # Convert to numpy array and average across segments
        mfccs = np.array(mfccs)
        mfccs = np.mean(mfccs, axis=0)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Add batch and channel dim

        # Make prediction
        predictions = model.predict(mfccs)
        predicted_index = np.argmax(predictions, axis=1)

        return predicted_index[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    try:
        # Load model and mapping
        model, genre_mapping = load_model_and_mapping()

        # Make prediction
        file_path = "BAK.wav"
        predicted_index = predict_genre(model, file_path)
        predicted_genre = genre_mapping[predicted_index]

        print(f"Predicted genre: {predicted_genre}")
    except Exception as e:
        print(f"Prediction failed: {e}")