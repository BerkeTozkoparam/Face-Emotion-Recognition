# Face Mesh and Emotion Recognition

A real-time face mesh detection and emotion recognition application using MediaPipe and OpenCV.

## Features

- Real-time face landmark detection
- Emotion recognition (Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral)
- Bounding box around detected faces
- FPS display
- Built with MediaPipe Tasks API

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BerkeTozkoparam/Face-Emotion-Recognition.git
   cd Face-Emotion-Recognition
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the model:
   - The `face_landmarker.task` model file is required but not included in the repo due to size.
   - Download it from [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) and place it in the project root.

## Usage

Run the application:
```bash
python main.py
```

- Press ESC to exit.
- Ensure camera permissions are granted on macOS.

## How It Works

- Uses MediaPipe's Face Landmarker to detect facial landmarks and blendshapes.
- Analyzes blendshapes to classify emotions based on facial expressions.
- Displays the detected emotion with confidence score.

## License

MIT License