from picamera2 import Picamera2

# Create a Picamera2 instance
camera = Picamera2()

# Configure the camera
config = camera.create_still_configuration()
camera.configure(config)

# Start the camera
camera.start()

# Capture an image and save it to a file
camera.capture_file("image.jpg")

# Stop the camera
camera.stop()