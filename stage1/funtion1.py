from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_inception_lstm_model(frames_per_video, frame_size, num_classes):
    # Load InceptionV3 model without the top layer
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*frame_size, 3))
    # Freeze the weights of InceptionV3
    base_model.trainable = False

    # Create the input layer
    video_input = Input(shape=(frames_per_video, *frame_size, 3))

    # Use TimeDistributed to wrap the Inception model so that each frame can be processed by InceptionV3
    encoded_frame_sequence = TimeDistributed(base_model)(video_input) # Apply InceptionV3 to each frame
    encoded_frame_sequence = TimeDistributed(GlobalAveragePooling2D())(encoded_frame_sequence) # Apply global average pooling to the output of InceptionV3

    # Apply the LSTM layer
    encoded_video = LSTM(256)(encoded_frame_sequence)  # Output dimension of the LSTM layer

    # Add fully connected and Dropout layers
    x = Dense(1024, activation='relu')(encoded_video)
    x = Dropout(0.5)(x)

    # Add the output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build the final model
    model = Model(inputs=video_input, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class VideoFrameGenerator(Sequence):
    def __init__(self, data_file, data_dir, batch_size, frames_per_video, frame_size, num_classes, shuffle=True):
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
        video_ids, labels = [], []
        with open(self.data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_id, _, label = line.strip().split(' ')
                video_ids.append(video_id)
                labels.append(int(label))
        return video_ids, labels

    def __len__(self):
        return int(np.floor(len(self.video_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        video_ids_temp = [self.video_ids[k] for k in indexes]
        X, y = self.data_generation(video_ids_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, video_ids_temp):
        # Initialize X and y
        X = np.empty((self.batch_size, self.frames_per_video, *self.frame_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, video_id in enumerate(video_ids_temp):
            frame_folder = os.path.join(self.data_dir, video_id)
            frame_paths = sorted(os.listdir(frame_folder))[:self.frames_per_video]
            for j, frame_path in enumerate(frame_paths):
                img_path = os.path.join(frame_folder, frame_path)
                img = load_img(img_path, target_size=self.frame_size)
                X[i, j,] = img_to_array(img) / 255.  # Normalize

            index_in_full_list = self.video_ids.index(video_id)  # Find the index of the current video_id in the global list
            y[i] = self.labels[index_in_full_list]  # Get label from the labels list

        return X, to_categorical(y, num_classes=self.num_classes)
