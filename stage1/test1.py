import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class VideoFrameGenerator(Sequence):
    def __init__(self, data_file, data_dir, batch_size, frames_per_video, frame_size, num_classes, shuffle=True):
        """
        Initializes the VideoFrameGenerator object.

        Args:
            data_file (str): Path to the file containing video data and labels.
            data_dir (str): Directory containing the video frames.
            batch_size (int): Number of samples per batch.
            frames_per_video (int): Number of frames to consider per video.
            frame_size (tuple): Size of each frame (height, width).
            num_classes (int): Number of classes or categories.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        self.data_file = data_file
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.video_ids, self.labels = self.load_video_labels()
        self.indexes = np.arange(len(self.video_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_video_labels(self):
        """
        Loads video IDs and corresponding labels from the data file.

        Returns:
            tuple: A tuple containing lists of video IDs and labels.
        """
        video_ids, labels = [], []
        with open(self.data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_id, _, label = line.strip().split(' ')
                video_ids.append(video_id)
                labels.append(int(label))
        return video_ids, labels

    def __len__(self):
        """
        Returns the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return int(np.floor(len(self.video_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: A tuple containing the input data and corresponding labels.
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        video_ids_temp = [self.video_ids[k] for k in indexes]
        X, y = self.data_generation(video_ids_temp)
        return X, y

    def on_epoch_end(self):
        """
        Shuffles the data at the end of each epoch.
        """
        self.indexes = np.arange(len(self.video_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, video_ids_temp):
        """
        Generates data containing batch_size samples.

        Args:
            video_ids_temp (list): List of video IDs for the current batch.

        Returns:
            tuple: A tuple containing the input data and corresponding labels.
        """
        X = np.empty((self.batch_size, self.frames_per_video, *self.frame_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, video_id in enumerate(video_ids_temp):
            frame_folder = os.path.join(self.data_dir, video_id)
            frame_paths = sorted(os.listdir(frame_folder))[:self.frames_per_video]
            for j, frame_path in enumerate(frame_paths):
                img = load_img(os.path.join(frame_folder, frame_path), target_size=self.frame_size)
                X[i, j,] = img_to_array(img) / 255.

            y[i] = self.labels[i]

        return X, to_categorical(y, num_classes=self.num_classes)

# Define the model
def create_model(frames_per_video, frame_size, num_classes):
    """
    Creates and compiles a deep learning model for video classification.

    Args:
        frames_per_video (int): Number of frames in each video.
        frame_size (tuple): Size of each frame (height, width).
        num_classes (int): Number of classes or categories.

    Returns:
        tensorflow.keras.Model: Compiled deep learning model.
    """
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(frames_per_video, *frame_size, 3)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(50),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def print_hi(name):
    """
    Prints a greeting message.

    Args:
        name (str): Name to be included in the greeting message.
    """
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    print_hi('PyCharm')
    # Parameter configuration
    data_dir = "E:/cityhunter/archive_2"
    train_file = "E:/cityhunter/train_videofolder.txt"
    val_file = "E:/cityhunter/val_videofolder.txt"
    batch_size = 16
    frames_per_video = 35
    frame_size = (176, 100)
    num_classes = 8  # Number of classes, adjust according to your actual dataset

    # Create data generator instances
    train_generator = VideoFrameGenerator(train_file, data_dir, batch_size, frames_per_video, frame_size, num_classes)
    val_generator = VideoFrameGenerator(val_file, data_dir, batch_size, frames_per_video, frame_size, num_classes)

    # Create the model
    model = create_model(frames_per_video, frame_size, num_classes)

    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=4)

    # Save the model
    model.save("gesture_recognition_model.h5")