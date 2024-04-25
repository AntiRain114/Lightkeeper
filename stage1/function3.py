def _bn_relu(inputs):
    """Batch Normalization -> ReLU activation"""
    norm = layers.BatchNormalization()(inputs)
    return layers.Activation("relu")(norm)

def _conv_bn_relu3D(filters, kernel_size, strides=(1, 1, 1), kernel_regularizer=regularizers.l2(1e-4)):
    """Build Conv3D -> BN -> ReLU block"""
    def f(inputs):
        conv = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                             kernel_initializer="he_normal", padding="same",
                             kernel_regularizer=kernel_regularizer)(inputs)
        return _bn_relu(conv)
    return f

def _bn_relu_conv3d(filters, kernel_size, strides=(1, 1, 1), kernel_regularizer=regularizers.l2(1e-4)):
    """Build BN -> ReLU -> Conv3D block"""
    def f(inputs):
        activation = _bn_relu(inputs)
        return layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                             use_bias=False, kernel_initializer="he_normal",
                             kernel_regularizer=kernel_regularizer)(activation)
    return f

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def _shortcut3d(input, residual):
    """Build shortcut connection for 3D residual block"""
    input_shape = tf.shape(input)
    residual_shape = tf.shape(residual)

    # Calculate strides
    stride_dim1 = input_shape[1] // residual_shape[1]
    stride_dim2 = input_shape[2] // residual_shape[2]
    stride_dim3 = input_shape[3] // residual_shape[3]
    equal_channels = tf.equal(input_shape[4], residual_shape[4])

    # Condition and adapt dimensions
    shortcut = tf.cond(
        tf.reduce_any([stride_dim1 != 1, stride_dim2 != 1, stride_dim3 != 1, not equal_channels]),
        lambda: layers.Conv3D(filters=residual_shape[4], kernel_size=(1, 1, 1),
                              strides=(stride_dim1, stride_dim2, stride_dim3),
                              padding="valid", use_bias=False,
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input),
        lambda: input
    )
    shortcut = layers.BatchNormalization()(shortcut)
    return layers.Add()([shortcut, residual])

def _residual_block3d(block_function, filters, repetitions, is_first_layer=False, kernel_regularizer=regularizers.l2(1e-4)):
    """Build repeated 3D residual blocks"""
    def f(inputs):
        for i in range(repetitions):
            init_strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2, 2)
            inputs = block_function(filters, init_strides, kernel_regularizer)(inputs)
        return inputs
    return f


def basic_block(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_regularizer=regularizers.l2(1e-4)):
    """Basic 3D residual block"""
    def f(inputs):
        conv = _bn_relu_conv3d(filters, kernel_size, strides=strides, kernel_regularizer=kernel_regularizer)(inputs)
        residual = _bn_relu_conv3d(filters, kernel_size, kernel_regularizer=kernel_regularizer)(conv)
        return _shortcut3d(inputs, residual)
    return f

def bottleneck(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_regularizer=regularizers.l2(1e-4)):
    """Bottleneck 3D residual block"""
    def f(inputs):
        conv1 = _bn_relu_conv3d(filters, (1, 1, 1), kernel_regularizer=kernel_regularizer)(inputs)
        conv2 = _bn_relu_conv3d(filters, kernel_size, strides=strides, kernel_regularizer=kernel_regularizer)(conv1)
        residual = _bn_relu_conv3d(filters * 4, (1, 1, 1), kernel_regularizer=kernel_regularizer)(conv2)
        return _shortcut3d(inputs, residual)
    return f


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
        """Load video IDs and corresponding labels from the data file"""
        video_ids, labels = [], []
        with open(self.data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_id, _, label = line.strip().split(' ')
                video_ids.append(video_id)
                labels.append(int(label))
        return video_ids, labels

    def __len__(self):
        """Return the number of batches in the dataset"""
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
        """Generate data containing batch_size samples"""
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

# Define the model
def create_model(frames_per_video, frame_size, num_classes):
    model = Sequential([
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), input_shape=(frames_per_video, *frame_size, 3)),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(256, return_sequences=False),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype=tf.float32)  # Ensure the last Dense layer uses float32 data type
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class Resnet3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor=1e-4, drop_rate=0):
        """Build 3D ResNet model"""
        inputs = tf.keras.Input(shape=input_shape)

        # Initial convolution and pooling layers
        x = _conv_bn_relu3D(64, (7, 7, 7), strides=(2, 2, 2), kernel_regularizer=regularizers.l2(reg_factor))(inputs)
        x = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)

        # Apply residual blocks
        filters = 64
        for i, r in enumerate(repetitions):
            x = _residual_block3d(block_fn, filters, r, is_first_layer=(i == 0),
                                  kernel_regularizer=regularizers.l2(reg_factor))(x)
            filters *= 2
            if drop_rate > 0:
                x = layers.Dropout(drop_rate)(x)

        # End with BN and ReLU
        x = _bn_relu(x)

        # Global average pooling and dense layer
        x = layers.GlobalAveragePooling3D()(x)
        outputs = layers.Dense(num_outputs, activation='softmax', kernel_regularizer=regularizers.l2(reg_factor))(x)

        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4, drop_rate=0):
        """Build ResNet3D-101 model"""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3], reg_factor, drop_rate)
