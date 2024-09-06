import numpy as np
import cv2 as cv
from tflite_runtime.interpreter import Interpreter
import os
import RPi.GPIO as GPIO
import time
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LED_PIN = 18

def load_and_preprocess_image(path):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (100, 100))
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict_gesture(image, interpreter):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_gesture = np.argmax(output_data)
    return predicted_gesture

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LED_PIN, GPIO.OUT)

def blink_led(frequency):
    global blink_flag
    delay = 1 / (2 * frequency) if frequency > 0 else 0
    while blink_flag:
        if frequency > 0:
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(delay)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.1)

def control_led_frequency(gesture_class):
    global blink_flag, blink_thread
    
    if blink_thread is not None:
        blink_flag = False
        blink_thread.join()

    frequencies = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]  # 0 Hz (off) and 9 different frequencies
    frequency = frequencies[gesture_class]
    
    print(f"LED blinking at {frequency} Hz")
    
    blink_flag = True
    blink_thread = threading.Thread(target=blink_led, args=(frequency,))
    blink_thread.start()

if __name__ == "__main__":
    setup_gpio()
    blink_flag = False
    blink_thread = None

    model_path = './model.tflite'
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    try:
        while True:
            image_path = './image.jpg'
            image = load_and_preprocess_image(image_path)
            gesture_class = predict_gesture(image, interpreter)
            print("Recognized Gesture Number:", gesture_class)
            control_led_frequency(gesture_class)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        blink_flag = False
        if blink_thread is not None:
            blink_thread.join()
        GPIO.cleanup()
        print("GPIO cleaned up")