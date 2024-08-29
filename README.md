# Theft Detection Using YOLOv8

![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0.1-blueviolet)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset%20Processing-orange)

## Project Overview

This project aims to develop a sophisticated theft detection system using the YOLOv8 object identification algorithm. The system is integrated into a Flask web application that allows real-time monitoring. The model is trained on a large dataset of 13,382 images, which were carefully annotated using Roboflow. The system is capable of identifying objects such as axes, knives, firearms, and other potential theft-related items in real-time.

## Features

- **Real-time Detection**: Detects theft-related objects in real-time using YOLOv8.
- **Web Interface**: Integrated with Flask for easy deployment and user interaction.
- **High Accuracy**: Achieved a precision-recall curve of 0.935 at mAP @ 0.5 after training for 150 epochs.
- **Extensive Dataset**: Trained on a diverse dataset with over 13,000 annotated images.
- **Scalable**: Designed to be scalable and robust, suitable for various security-sensitive environments.

## Tech Stack

- **Python**: Core language for implementing the project.
- **YOLOv8**: Object detection algorithm used for training the model.
- **Flask**: Web framework used to create the user interface.
- **Roboflow**: Tool used for dataset annotation and augmentation.
- **Google Colab Pro**: Used for training the model with GPU support.
- **OpenCV**: Utilized for image processing tasks.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jaswanth-mjy/TheftDetectionSystem.git
   cd TheftDetectionSystem
   ```

2. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 Weights**
   - Download the `best.pt` file from [Roboflow](https://roboflow.com/) and place it in the root directory of the project.

4. **Set Up Environment Variables**
   - Create a `.env` file and add the following:
     ```
     ROBOFLOW_API_KEY=your_api_key
     ```

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Web Interface**
   - Open your web browser and go to `http://127.0.0.1:5000/` to interact with the application.

## Usage

- **Upload Image/Video**: Upload images or videos through the web interface to detect objects.
- **Real-time Monitoring**: Use the real-time monitoring feature to watch live camera feeds.
- **View Results**: The application displays detected objects with bounding boxes and labels.

## Code Implementation

### YOLOv8 Model Training

```python
from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.0.20
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

import zipfile

# Path to the zip file
zip_file_path = '/content/drive/MyDrive/Jaswanth_tharun_sdp_project1.v1i.yolov8.zip'
# Directory to extract the contents
extract_dir = '/content/drive/MyDrive/sdp_dataset/'

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Initialize YOLOv8 model
from ultralytics import YOLO
from IPython.display import display, Image

%cd /content/drive/MyDrive/sdp_dataset
!ls

# Train the model
!yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=150 imgsz=640 plots=True
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

# Perform prediction
%cd /content/drive/MyDrive/sdp_dataset
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=test/images

%cd /content/drive/MyDrive/
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.77 source=/content/drive/MyDrive/testvideo.mp4 save=True
```

### Flask Deployment

```python
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from ultralytics import YOLO

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

# Initialize YOLOv8 model
model = YOLO(weights='best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img_path = '/path/to/save/uploads/' + file.filename
        file.save(img_path)
        
        # Perform inference with YOLOv8 model
        results = model(img_path)
        predictions = results.pred  # Process results
        
        return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run()
```

## Model Training

The YOLOv8 model was trained using the following steps:

1. **Data Collection**: Images were collected and annotated using Roboflow.
2. **Data Preprocessing**: The dataset was preprocessed with techniques like flipping, rotating, and resizing.
3. **Model Training**: The model was trained for 150 epochs on Google Colab Pro with Tesla T4 GPU.
4. **Evaluation**: The model achieved a high precision-recall score, indicating strong performance across all classes.

## Future Enhancements

- **Add More Classes**: Extend the model to detect more types of objects.
- **Improve UI**: Enhance the web interface for a better user experience.
- **Mobile Integration**: Develop a mobile application for on-the-go monitoring.

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- **[Dr. E. Ajith Jubilson](https://www.linkedin.com/in/ajith-jubilson-a5b03242/?originalSubdomain=in)**: Project Supervisor.
- **Roboflow**: For providing the tools for dataset management and annotation.
- **VIT University**: For supporting this project as part of the B.Tech curriculum.

---
