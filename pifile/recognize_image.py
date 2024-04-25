import numpy as np
import cv2 as cv
from tflite_runtime.interpreter import Interpreter
import RPi.GPIO as GPIO
import time

# Set TensorFlow log level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up GPIO for LED control
LED_PIN = 18  # Replace with the appropriate GPIO pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Create a PWM instance for LED brightness control
pwm = GPIO.PWM(LED_PIN, 100)  # Frequency: 100 Hz
pwm.start(0)  # Start PWM with 0% duty cycle

def load_and_preprocess_image(frame):
    # Convert to RGB format (OpenCV defaults to BGR)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Resize the image
    image = cv.resize(image, (100, 100))
    # Normalize the image data
    image = image / 255.0
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

def set_led_brightness(gesture_class):
    if 0 <= gesture_class <= 2:
        pwm.ChangeDutyCycle(0)  # Turn off the LED
    elif 3 <= gesture_class <= 5:
        pwm.ChangeDutyCycle(50)  # Set normal brightness (50% duty cycle)
    elif 6 <= gesture_class <= 9:
        pwm.ChangeDutyCycle(100)  # Set brighter brightness (100% duty cycle)

if __name__ == "__main__":
    # Path to the model
    model_path = './model.tflite'
    
    # Load the TFLite model and allocate tensors
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Initialize the camera
    cap = cv.VideoCapture(0)
    
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame from the camera")
            break
        
        # Load and preprocess the image
        image = load_and_preprocess_image(frame)
        
        # Predict the gesture
        gesture_class = predict_gesture(image, interpreter)
        
        # Print the recognized gesture number
        print("Recognized Gesture Number:", gesture_class)
        
        # Set LED brightness based on the recognized gesture
        set_led_brightness(gesture_class)
        
        # Wait for 1 second before the next iteration
        time.sleep(1)
    
    # Release the camera and clean up
    cap.release()
    pwm.stop()
    GPIO.cleanup()