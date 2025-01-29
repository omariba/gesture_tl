from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("gesture_model.h5")
class_labels = {i: label for i, label in enumerate(["Church","Enough/Satisfied","Friend","Love","Me","Mosque","Seat","Temple","You"])}  # Update with your class names

# Initialize webcam
cap = cv2.VideoCapture(0)

# Preprocess frame for model input
def preprocess_frame(frame, img_size=224):
    resized_frame = cv2.resize(frame, (img_size, img_size))
    normalized_frame = resized_frame / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=0)
    return expanded_frame

# Generate webcam frames
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)  # Mirror the image
            processed_frame = preprocess_frame(frame)
            predictions = model.predict(processed_frame)
            predicted_class = np.argmax(predictions, axis=1)[0]
            gesture_text = class_labels.get(predicted_class, "Unknown Gesture")

            # Overlay prediction on the frame
            cv2.putText(frame, f"Gesture: {gesture_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame to be sent to the web page
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
