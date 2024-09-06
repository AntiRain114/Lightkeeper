from picamera2 import Picamera2
from PIL import Image
import time

def capture_image_with_autofocus(width, height):
    # Initialize the camera
    camera = Picamera2()
    
    # Configure the camera for still capture with the specified resolution
    config = camera.create_still_configuration(main={"size": (width, height)})
    camera.configure(config)
    
    # Start the camera
    camera.start()
    
    # Allow time for the camera to adjust
    time.sleep(2)
    
    # Trigger autofocus
    camera.set_controls({"AfMode": 1, "AfTrigger": 0})
    
    # Wait for autofocus to complete
    time.sleep(2)
    
    # Capture the image
    image_path = 'captured_image.jpg'
    camera.capture_file(image_path)
    
    # Stop the camera
    camera.stop()
    
    print(f"Image captured with autofocus and saved as {image_path}")
    return image_path

def rotate_and_save_image(input_path, output_path):
    # Open the image
    image = Image.open(input_path)
    
    # Rotate the image 180 degrees
    rotated_image = image.rotate(180)
    
    # Save the rotated image
    rotated_image.save(output_path)
    print(f"Rotated image saved as {output_path}")

if __name__ == "__main__":
    # Set the desired image resolution (150's multiple)
    width = height = 2550
    
    # Capture image with autofocus
    captured_image_path = capture_image_with_autofocus(width, height)
    
    # Rotate and save the image
    rotated_image_path = 'test_image.jpg'
    rotate_and_save_image(captured_image_path, rotated_image_path)