# Silence

## 1. Introduction

**Silence** is a real-time sign-language recognition application that uses a webcam to recognize American Sign Language (ASL) hand gestures. It converts recognized ASL letters into a text buffer, suggests matching English words, supports numbers and custom signs, and provides a delete gesture for correcting text.

The system combines:

- MediaPipe 21-point hand landmark detection
- A 26-class PyTorch model for ASL letters A-Z
- A custom sign classifier for user-defined vocabulary
- An SVM-based thumbs-up detector for word confirmation
- An offline English dictionary and local confirmation feedback
- OpenCV for webcam access and live visual feedback

This project is designed to run locally on Windows and does not require a cloud API for recognition.

### 1.1 Objectives

- Recognize static and motion-based ASL letter gestures from webcam input
- Convert recognized letters into a readable text sequence
- Suggest and confirm likely English words
- Support number signs and application-defined custom gestures
- Allow users to collect their own data and retrain the classifiers
- Provide immediate, on-screen status and correction feedback

---

## 2. System Overview

The live recognition pipeline is:

```text
Webcam frame
    -> MediaPipe Hand Landmarker
    -> 21 hand landmarks (x, y, z)
    -> 15-frame pose and motion sequence
    -> Feature normalization and geometric feature extraction
    -> ASL letter model
    -> Custom sign classifier
    -> Number-sign resolver
    -> Recognition smoothing and text buffer
    -> Word suggestion and thumbs-up confirmation
```

The application collects 15 consecutive pose samples at approximately 15 frames per second. A pose is represented by normalized 3D joint coordinates, the wrist position, and frame-to-frame velocity features.

---

## 3. Model Architecture

### 3.1 ASL Letter Sequence Model

The ASL recognition model classifies 26 letters and is selected automatically by `train.py` from two architectures:

#### Bidirectional GRU

- Input feature size: 194 values per frame after geometric enrichment
- Two bidirectional GRU layers
- 128 hidden units per layer
- 25% recurrent dropout
- Batch normalization at the input
- Average and maximum temporal pooling
- Layer normalization, 30% dropout, and a 26-class linear classifier

#### Temporal CNN

- Input feature size: 194 values per frame after geometric enrichment
- 1D temporal convolution projection
- Dilated 1D convolutions with kernel size 3
- Dilation rates of 1, 2, and 4
- GELU activations, batch normalization, and dropout
- Adaptive average pooling over three temporal segments
- Adaptive maximum pooling
- Layer normalization and a 26-class linear classifier

The training script compares both architectures using unseen collection sessions and saves the better-performing checkpoint to `asl_motion_model.pth`.

### 3.2 Input Features

Each of the 15 frames contains:

| Feature group | Size | Description |
|---|---:|---|
| Normalized 3D hand joints | 63 | 21 MediaPipe landmarks relative to the wrist |
| Wrist position | 3 | Wrist world/camera position |
| De-amplified velocities | 66 | Frame-to-frame pose movement |
| Joint distances | 31 | 3D distances between selected joints |
| Bone angles | 15 | Finger curl and inter-finger spread angles |
| Tip ratios | 5 | Fingertip radius normalized by palm size |
| Thumb ratios | 4 | Thumb-tip height relative to finger joints |
| Pinch indicator | 1 | Thumb-to-index contact score |
| Motion trajectory | 6 | Cumulative index and pinky motion odometers |

The pose and velocity data is enriched in `sequence_model.py` to improve recognition of visually or geometrically similar signs.

### 3.3 Custom Sign Model

Custom signs use the same 15-frame gesture representation. `custom_train.py` trains either:

- A calibrated RBF SVM for two or more custom labels
- A one-class RBF SVM when only one custom sign has been recorded

Features are flattened and standardized before classification. The trained classifier is stored as `custom_sign_model.joblib`.

### 3.4 Thumbs-Up Confirmation Model

The thumbs-up detector uses 66 static pose features:

- StandardScaler for feature normalization
- RBF Support Vector Machine
- `CalibratedClassifierCV` for probability estimates
- Balanced class weighting

The model is stored as `thumb_gesture_model.joblib`. If no trained model is available, the application can fall back to a basic MediaPipe hand-pose rule.

### 3.5 Word Prediction

Word prediction is dictionary-based rather than neural. `WordPredictor`:

1. Loads words from `english_words.txt`
2. Filters words that begin with the current letter sequence
3. Prioritizes learned confirmed words
4. Applies a common-word ranking
5. Falls back to shortest-word and alphabetical ordering

Accepted confirmations are appended to `word_feedback.jsonl`. The feedback raises the ranking of previously accepted words for future suggestions.

---

## 4. How the Project Is Built

### 4.1 Project Structure

| File | Purpose |
|---|---|
| `webcam.py` | Main real-time application |
| `hand_tracking.py` | MediaPipe hand-landmark setup and drawing |
| `sequence_dataset.py` | Dataset loading and gesture feature conversion |
| `sequence_model.py` | Geometric feature enrichment and PyTorch architectures |
| `train.py` | Trains and validates the 26-letter model |
| `predict.py` | Loads and runs the trained ASL model |
| `collect_data.py` | Collects ASL sequences and thumbs-up samples |
| `evaluate.py` | Reports validation accuracy and confusion pairs |
| `custom_signs.py` | Stores and matches custom sign examples |
| `custom_train.py` | Trains the custom-sign classifier |
| `thumb_gesture.py` | Trains and loads thumbs-up classification |
| `word_prediction.py` | Suggests words and records confirmations |
| `train_colab.ipynb` | ASL training workflow for Google Colab |

### 4.2 Trained Model and Data Files

| File | Description |
|---|---|
| `hand_landmarker.task` | MediaPipe hand landmark model |
| `asl_motion_model.pth` | Trained 26-letter PyTorch model |
| `asl_motion_sequence_dataset.csv` | Recorded ASL gesture sequences |
| `custom_sign_model.joblib` | Trained custom sign classifier |
| `custom_signs.json` | Recorded custom sign examples |
| `thumb_gesture_model.joblib` | Trained thumbs-up classifier |
| `thumb_gesture_dataset.csv` | Collected thumbs-up and negative samples |
| `english_words.txt` | Offline English dictionary |
| `word_feedback.jsonl` | Locally recorded word confirmations |

### 4.3 ASL Model Training

`train.py` performs session-separated validation:

1. Loads sequences from `asl_motion_sequence_dataset.csv`
2. Checks that all 26 letters are represented
3. Checks for at least two recording sessions
4. Splits data without mixing sessions across train and validation sets
5. Applies temporal shifts, rotations, scaling, and sensor-jitter augmentation
6. Trains GRU and Temporal CNN architectures for 65 epochs
7. Calculates class weights for uneven letter counts
8. Selects the architecture with the best validation accuracy
9. Saves the best checkpoint to `asl_motion_model.pth`

Training automatically selects CUDA when available and otherwise uses the CPU.

### 4.4 Custom Sign Training

A custom sign is represented by 10 repetitions of a 15-frame sequence. Repetitions are combined with pose and velocity features, then used to train the custom classifier. If fewer signs are available, the system first performs nearest-example matching before a trained model is ready.

---

## 5. Library and Tool Reference

| Library | Function in this project |
|---|---|
| `MediaPipe` | Detects one hand and returns 21 3D landmarks |
| `OpenCV` | Opens the webcam, renders video, and handles keyboard input |
| `PyTorch` | Defines, trains, validates, and runs GRU/CNN ASL models |
| `NumPy` | Builds arrays and calculates normalized landmark features |
| `scikit-learn` | Provides session splitting, SVM classifiers, calibration, and scaling |
| `joblib` | Serializes custom-sign and thumbs-up models |
| `Pandas` | Available for data processing in the project environment |
| `Matplotlib` | Available for plots or diagnostics |
| `Pillow` | Available for image processing support |
| `SciPy` | Provides scientific and numerical support for machine learning tools |
| `Python standard library` | Handles CSV, JSON, paths, timing, counters, and application data |

---

## 6. Features

- Live ASL letter recognition for A-Z
- 15-frame motion sequence capture
- Probability smoothing over five prediction windows
- Stable-letter commitment after 0.6 seconds
- Offline English word suggestions
- Thumbs-up word confirmation
- 5-second suggestion decision window
- Number-sign recognition with labels 0-9
- Custom sign registration and automatic retraining
- Custom delete gesture
- Local confirmation feedback through `word_feedback.jsonl`
- Screen-scrolling recognized text for long input
- Session-separated ASL validation
- Automatic CPU or CUDA training
- Google Colab training support

---

## 7. Installation and Setup

### 7.1 Requirements

- Windows
- Python 3
- A webcam
- A terminal/PowerShell window

### 7.2 Create a Virtual Environment

```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

If script execution is disabled, use Command Prompt instead:

```bat
myvenv\Scripts\activate.bat
```

### 7.3 Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `hand_landmarker.task` remains in the project root.

---

## 8. Using the Application

### 8.1 Start Live Recognition

```powershell
python webcam.py
```

Show one hand to the webcam and hold or perform the sign.

### 8.2 Controls

| Key | Function |
|---|---|
| `q` | Quit live recognition |
| `c` | Begin recording a custom sign |

### 8.3 Letter and Word Workflow

1. The application recognizes a stable ASL letter.
2. The letter is added to the current word after its stability period.
3. The side panel displays up to one matching English word suggestion.
4. Hold a thumbs-up for at least 0.35 seconds to confirm the suggestion.
5. Sign the next letter within 5 seconds to update the suggestion.
6. Wait 5 seconds without an action to keep the spelled letters as text.

### 8.4 Number Signs

Press `c` in live recognition and enter a label from `0` through `9`.

- Record 10 repetitions for each number sign.
- If a number sign resembles an ASL letter, the number is held for 1.5 seconds before it is committed.
- Holding the gesture longer resolves it as a number.
- Performing a different gesture resolves the result as the detected letter.

### 8.5 Custom Signs

1. Run `python webcam.py`
2. Press `c`
3. Enter a label in the terminal
4. Hold/reset your hand during the prompts
5. Record 10 repetitions
6. The custom classifier retrains automatically

Run the training command manually if automatic training was interrupted:

```powershell
python custom_train.py
```

Record at least five examples per custom sign before manually training multiple signs.

### 8.6 Delete Gesture

Record a custom sign with the exact label `delete`, then perform it while using live recognition. It removes:

- The latest letter in the current word, or
- The latest confirmed word or number when no letters are in the current word

---

## 9. Data Collection

### 9.1 Collect ASL Letter Sequences

```powershell
python collect_data.py
```

Press an `A-Z` key to record that letter. Static letters should be held still, while moving signs should be performed across the 15-frame capture window.

Record every letter in at least two separate collection sessions. Restart `collect_data.py` to create a new session ID.

### 9.2 Collect Thumbs-Up Data

In `collect_data.py`:

| Key | Capture |
|---|---|
| `Space` | Thumbs-up example |
| `0` | Negative/other pose example |
| `/` | Quit |

Capture at least three separate examples of each pose. The detector retrains automatically after enough samples are collected.

### 9.3 Train the ASL Model

```powershell
python train.py
```

Evaluate the saved model against its unseen-session validation split:

```powershell
python evaluate.py
```

For Colab:

1. Open `train_colab.ipynb`
2. Select a GPU runtime
3. Upload `train.py`, `sequence_dataset.py`, `sequence_model.py`, and `asl_motion_sequence_dataset.csv` when prompted
4. Run all cells
5. Download `asl_motion_model.pth`

---

## 10. Testing

The repository includes scripts for testing individual components:

```powershell
python test_webcam.py
python test_custom_signs.py
python test_custom_train.py
python test_thumb_gesture.py
python test_word_prediction.py
```

Run tests individually so that each script can verify its relevant webcam, model, dictionary, or data behavior.

---

## 11. Output and Saved Files

The application creates or updates:

- `custom_signs.json` for custom sign examples
- `custom_sign_model.joblib` after custom-sign training
- `word_feedback.jsonl` after accepted word confirmations
- `thumb_gesture_dataset.csv` during thumbs-up collection
- `thumb_gesture_model.joblib` after thumbs-up training
- `asl_motion_sequence_dataset.csv` during ASL sequence collection
- `asl_motion_model.pth` after ASL training

Generated training datasets and local models should be managed carefully because they may contain application-specific data and capture patterns.

---

## 12. Limitations

- Recognition quality depends on lighting, webcam position, hand size, and user consistency
- The project currently recognizes one hand
- ASL signs that are highly similar visually may require more balanced training samples
- Number/letter ambiguity is resolved using user hold timing and custom-sign conflicts
- Word suggestions use a static English dictionary and confirmation ranking
- The included pretrained models reflect the available dataset and may require retraining for better personal or regional performance

---

## 13. Dictionary License

The offline English word list in `english_words.txt` is derived from:

- [dwyl/english-words](https://github.com/dwyl/english-words)

The source dedicates the word list to the public domain. The complete license text is included in `WORDLIST_LICENSE.md`.
