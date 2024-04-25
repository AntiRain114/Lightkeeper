# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import tensorflow as tf
import os
import numpy as np
from skimage import io
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# 读取标签文件
label_df = pd.read_csv('E:\\cityhunter\\jester-v1-labels.csv', header=None)
# 创建标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(label_df[0])

def load_dataset(csv_file, label_encoder, data_dir='E:\\cityhunter\\archive_2'):
    df = pd.read_csv(csv_file, sep=';', header=None)
    file_paths = []
    labels = []

    for _, row in df.iterrows():
        video_id, label = row
        label_encoded = label_encoder.transform([label])[0]
        video_path = os.path.join(data_dir, str(video_id))  # 确保 video_id 是字符串
        file_paths.append(video_path)
        labels.append(label_encoded)

    return file_paths, labels

# 加载数据集
train_file_paths, train_labels = load_dataset('E:\\cityhunter\\jester-v1-train.csv', label_encoder)
val_file_paths, val_labels = load_dataset('E:\\cityhunter\\jester-v1-validation.csv', label_encoder)

from skimage.transform import resize

def load_video_frames(video_path, num_frames=35, target_height=100, target_width=176):
    frame_paths = sorted([os.path.join(video_path, frame) for frame in os.listdir(video_path)])
    frames = [io.imread(frame_path) for frame_path in frame_paths]

    # 以下保持不变
    processed_frames = np.array([resize(frame, (target_height, target_width), mode='constant') for frame in frames])
    processed_frames = processed_frames.astype('float32') / 255.0  # 归一化
    return processed_frames



def make_dataset(file_paths, labels, batch_size=8):
    def generator():
        for video_path, label in zip(file_paths, labels):
            video_frames = load_video_frames(video_path)
            yield video_frames, label

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32), output_shapes=((None, None, None, 3), ()))
    dataset = dataset.batch(batch_size)
    return dataset


train_dataset = make_dataset(train_file_paths, train_labels)
val_dataset = make_dataset(val_file_paths, val_labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense

# 定义模型参数
num_classes = 27
frame_height, frame_width = 100, 176  # 请根据你的数据调整
channels = 3
num_frames = 35  # 请根据你的数据调整

# 构建模型
model = Sequential([
    # TimeDistributed Conv2D layers
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(num_frames, frame_height, frame_width, channels)),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),

    # LSTM layer
    LSTM(64),

    # Dense layers
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    model.save('model1')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
