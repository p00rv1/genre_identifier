# 🎵 Genre Identifier

Automatically identify the genre of an audio file using machine learning.

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 📌 Overview

A tool that predicts the genre (e.g., Rock, Jazz, Hip-hop) of **.wav** or **.mp3** audio files using a trained model.

---

## ⚙️ Features

- 🎶 Supports `.wav` and `.mp3` formats  
- Uses a convolutional neural network (CNN) on spectrograms or MFCCs  
- Easy to use: just provide the audio path  
- 🎛️ Outputs genre label with confidence score  

---

## 🛠️ Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/p00rv1/genre_identifier.git
    cd genre_identifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

```bash
python predict_genre.py --input path/to/song.wav
