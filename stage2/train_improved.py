import tensorflow as tf
import os
import numpy as np
import cv2 as cv
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers, Model, Input
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_files(file_dir):
    image_list = []
    label_list = []
    for class_id in range(10):  # Assuming 10 classes (0-9)
        class_path = os.path.join(file_dir, str(class_id))
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image_list.append(image_path)
            label_list.append(class_id)

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = [int(i) for i in temp[:, 1]]
    return image_list, label_list


def preprocess_image(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [100, 100])
    image = image / 255.0  # normalize to [0,1] range
    return image, label


def data_augmentation(image, label):
    # Random flip left-right
    image = tf.image.random_flip_left_right(image)

    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation adjustment
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Random hue adjustment
    image = tf.image.random_hue(image, max_delta=0.1)

    # Ensure the image values are still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def build_improved_model(input_shape=(100, 100, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr *= 0.1
    if epoch > 20:
        lr *= 0.1
    return lr


def train_model(train_data, val_data):
    model = build_improved_model()
    model.summary()

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    checkpoint_filepath = "./model_save/gestureModel"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        train_data,
        epochs=40,  # Increased epochs, early stopping will prevent overfitting
        validation_data=val_data,
        callbacks=[model_checkpoint_callback, early_stopping, lr_scheduler]
    )

    model.save('./model/gestureModel_final')
    print("Model saved successfully")
    return model, history
def k_fold_cross_validation(image_list, label_list, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_list)):
        print(f"Training on fold {fold+1}/{n_splits}")

        # 准备这个折叠的数据
        train_images = [image_list[i] for i in train_idx]
        train_labels = [label_list[i] for i in train_idx]
        val_images = [image_list[i] for i in val_idx]
        val_labels = [label_list[i] for i in val_idx]

        # 创建数据集
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

        # 应用预处理和数据增强
        train_ds = train_ds.map(preprocess_image).map(data_augmentation)
        val_ds = val_ds.map(preprocess_image)

        # 批处理和预取
        BATCH_SIZE = 32
        train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # 训练模型
        model, history = train_model(train_ds, val_ds)

        # 保存这个折叠的结果
        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_loss': history.history['val_loss'][-1]
        })

        # 可选：保存每个折叠的模型
        model.save(f'./model/gestureModel_fold_{fold+1}')

    return fold_results

if __name__ == "__main__":
    train_dir = 'train_gesture_data'
    image_list, label_list = get_files(train_dir)

    # Split data into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_list, label_list, test_size=0.2, random_state=42)

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    # Apply preprocessing and augmentation
    train_ds = train_ds.map(preprocess_image).map(data_augmentation)
    val_ds = val_ds.map(preprocess_image)

    # Batch and prefetch
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 32

    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Train the model
    model, history = train_model(train_ds, val_ds)

    # 执行k折交叉验证
    results = k_fold_cross_validation(image_list, label_list, n_splits=5)

    # 打印结果
    for result in results:
        print(
            f"Fold {result['fold']}: Validation Accuracy = {result['val_accuracy']:.4f}, Validation Loss = {result['val_loss']:.4f}")

    # 计算平均性能
    avg_accuracy = np.mean([r['val_accuracy'] for r in results])
    avg_loss = np.mean([r['val_loss'] for r in results])
    print(f"\nAverage Validation Accuracy: {avg_accuracy:.4f}")
    print(f"Average Validation Loss: {avg_loss:.4f}")

    # 可选：绘制每个折叠的性能
    import matplotlib.pyplot as plt

    accuracies = [r['val_accuracy'] for r in results]
    losses = [r['val_loss'] for r in results]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 6), accuracies)
    plt.title('Validation Accuracy per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.bar(range(1, 6), losses)
    plt.title('Validation Loss per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('k_fold_results.png')
    plt.show()

    print("K-fold cross-validation completed. Results saved in k_fold_results.png")