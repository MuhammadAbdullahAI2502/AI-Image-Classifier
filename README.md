# AI Image Classifier & Object Detector

A versatile web app for real-time image classification and object detection. This project provides two interfaces: a user-friendly web application built with Streamlit and a REST API built with Flask.

## Features

- **Image Classification:** Classify images using a pre-trained ResNet18 model on the ImageNet dataset.
- **Object Detection:** Detect objects in images using a pre-trained Faster R-CNN model on the COCO dataset.
- **Dual Interface:** 
    - **Streamlit Web App:** An interactive and easy-to-use interface for uploading images and viewing results.
    - **Flask API:** A REST API for programmatic access to the classification and detection models.
- **Real-time Inference:** Performs classification and detection in real-time.

## Dependencies

The project uses the following libraries:

- streamlit
- torch
- torchvision
- opencv-python-headless
- numpy
- Pillow
- requests
- Flask

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI-Image-Classifer.git
    cd AI-Image-Classifer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Streamlit Web App

To run the Streamlit web application:

```bash
streamlit run app.py
```

Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### Flask API

To run the Flask REST API:

```bash
python api.py
```

The API will be running at `http://localhost:5000`.

## API Endpoints

### 1. Image Classification

- **Endpoint:** `/predict/classify`
- **Method:** `POST`
- **Form Data:** `file`: The image file to be classified.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        [
            {
                "class": "giant panda",
                "probability": 0.99
            },
            {
                "class": "lesser panda",
                "probability": 0.003
            }
        ]
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`

### 2. Object Detection

- **Endpoint:** `/predict/detect`
- **Method:** `POST`
- **Form Data:** `file`: The image file for object detection.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        [
            {
                "label": "person",
                "confidence": 0.98,
                "box": [x1, y1, x2, y2]
            },
            {
                "label": "car",
                "confidence": 0.92,
                "box": [x1, y1, x2, y2]
            }
        ]
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`
