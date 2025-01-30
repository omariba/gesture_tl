# gesture_tl
A machine learning-powered gesture-to-text translation system that recognizes hand gestures and converts them into text using computer vision and deep learning.

**Project Overview**

GestureTL is designed to bridge communication gaps for sign language users by leveraging:

 - Deep Learning (MobileNetV2) for real-time gesture recognition.
 - Computer Vision (OpenCV, TensorFlow/Keras) for processing gesture images.
 - Flask & Web UI for an intuitive user interface to capture and translate gestures.

**Features:**

 ✅ Real-time gesture recognition via webcam input.
 
 ✅ Trained on the Kenyan Sign Language dataset.
 
 ✅ Transfer learning with MobileNetV2 for high accuracy.
 
 ✅ Web-based user interface for easy interaction.



 🛠️ Tech Stack

 - Backend: Python, TensorFlow/Keras, Flask

 - Frontend: HTML, CSS, JavaScript

 - Machine Learning: MobileNetV2, OpenCV, NumPy, Pandas

 - Development Tools: VS Code, Git



📂 Dataset & Preprocessing

This project uses the Kenyan Sign Language dataset, which includes labeled images of various gestures.

🔹 Preprocessing Steps:

 - Image resizing to 224x224 pixels.

 - Normalization (pixel values scaled to [0,1]).

 - Splitting into training (80%) and validation (20%) sets.


