import os
import librosa
import math
import json

DATASET_PATH = "genres_original"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)  # Convert to integer
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            dirpath_components = os.path.normpath(dirpath).split(os.sep)
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            for f in filenames:
                if not f.endswith('.wav'):  # Skip non-audio files
                    continue

                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment  # Fixed calculation

                        # Ensure we don't go beyond the signal length
                        if finish_sample > len(signal):
                            continue

                        segment = signal[start_sample:finish_sample]

                        # Extract MFCCs with proper keyword arguments
                        mfcc = librosa.feature.mfcc(
                            y=segment,
                            sr=sr,
                            n_fft=n_fft,
                            n_mfcc=n_mfcc,
                            hop_length=hop_length
                        )
                        mfcc = mfcc.T

                        if len(mfcc) == expected_num_mfcc_vector_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)
                            print("{}, segment:{}".format(file_path, s))
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)