import numpy as np
import cv2 as cv
import tensorflow as tf

# Setting the TensorFlow log level
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_preprocess_image(path):
    # Read the image
    image = cv.imread(path)
    # Convert to RGB format (OpenCV defaults to BGR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Apply Gaussian Blur to reduce noise
    image = cv.GaussianBlur(image, (5, 5), 0)
    # Convert to grayscale if the model is trained on grayscale images
    # image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # Resize the image
    image = cv.resize(image, (100, 100))
    # Normalize the image data
    image = image / 255.0
    # Histogram Equalization to enhance the contrast of the image
    # For grayscale, use this:
    # image = cv.equalizeHist(image)
    # For color images, apply to each channel:
    cv.imwrite('processed_image1.jpg', image * 255)  # Multiply by 255 to convert back to 0-255 scale
    cv.imwrite('processed_image2.jpg', image )  # Multiply by 255 to convert back to 0-255 scale

    # Add a dimension to match the model's input requirements
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


def predict_gesture(image, interpreter):
    # Get input layer details and set input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the model
    interpreter.invoke()

    # Get output layer details and result
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Find the index of the highest probability
    predicted_gesture = np.argmax(output_data)
    return predicted_gesture


if __name__ == "__main__":
    # Path to the model
    model_path = './model.tflite'

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Path to the image file
    image_path = './1516.jpg'

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Predict the gesture
    gesture_class = predict_gesture(image, interpreter)

    # Print the recognized gesture number
    print("Recognized Gesture Number:", gesture_class)
