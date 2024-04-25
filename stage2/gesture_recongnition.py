import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import csv
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_image(im):
    image = cv.resize(im, (100, 100))
    # print("image.shape", len(image))
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    image2 = image.reshape(-1, 100, 100,  3)
    image3 = tf.cast(image2 / 255.0, tf.float32)
    return image3


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[x1:x2, y1:y2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return dst


# Get images, put them into corresponding lists, and attach labels to them, then put them into the label list
def get_files(file_dir):
    # Lists to store the image categories and labels: Class 0
    list_0 = []
    label_0 = []
    # Lists to store the image categories and labels: Class 1
    list_1 = []
    label_1 = []
    # Lists to store the image categories and labels: Class 2
    list_2 = []
    label_2 = []
    # Lists to store the image categories and labels: Class 3
    list_3 = []
    label_3 = []
    # Lists to store the image categories and labels: Class 4
    list_4 = []
    label_4 = []
    # Lists to store the image categories and labels: Class 5
    list_5 = []
    label_5 = []
    # Lists to store the image categories and labels: Class 6
    list_6 = []
    label_6 = []
    # Lists to store the image categories and labels: Class 6
    list_7 = []
    label_7 = []
    # Lists to store the image categories and labels: Class 8
    list_8 = []
    label_8 = []
    # Lists to store the image categories and labels: Class 9
    list_9 = []
    label_9 = []

    for file in os.listdir(file_dir):  # Get all the filenames under the file_dir path
        # print(file)
        # Concatenate to get the image file path
        image_file_path = os.path.join(file_dir, file)
        for image_name in os.listdir(image_file_path):
            # print('image_name',image_name)
            # Complete path of the image
            image_name_path = os.path.join(image_file_path, image_name)
            # print('image_name_path',image_name_path)
            # Put the image into the corresponding list
            if image_file_path[-1:] == '0':
                list_0.append(image_name_path)
                label_0.append(0)
            elif image_file_path[-1:] == '1':
                list_1.append(image_name_path)
                label_1.append(1)
            elif image_file_path[-1:] == '2':
                list_2.append(image_name_path)
                label_2.append(2)
            elif image_file_path[-1:] == '3':
                list_3.append(image_name_path)
                label_3.append(3)
            elif image_file_path[-1:] == '4':
                list_3.append(image_name_path)
                label_3.append(4)
            elif image_file_path[-1:] == '5':
                list_3.append(image_name_path)
                label_3.append(5)
            elif image_file_path[-1:] == '6':
                list_3.append(image_name_path)
                label_3.append(6)
            elif image_file_path[-1:] == '7':
                list_3.append(image_name_path)
                label_3.append(7)
            elif image_file_path[-1:] == '8':
                list_3.append(image_name_path)
                label_3.append(8)
            else:
                list_4.append(image_name_path)
                label_4.append(9)

    # Merge the data
    image_list = np.hstack((list_0, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9))
    label_list = np.hstack((label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9))
    # Shuffle the data
    print("imagename = ", image_list[:10])
    print("labelname = ", label_list[:10])

    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # Transpose
    np.random.shuffle(temp)

    # Convert all images and labels into lists
    image_list = list(temp[:, 0])
    image_list = [i for i in image_list]
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    # print(image_list)
    # print(label_list)
    return image_list, label_list


def get_tensor(image_list, label_list):
    ims = []
    for image in image_list:
        # Read the image under the path
        x = tf.io.read_file(image)
        # Map the path to the image, 3 channels
        x = tf.image.decode_jpeg(x, channels=3)
        # Resize the image
        x = tf.image.resize(x, [32, 32])
        # Put the image into the list
        ims.append(x)
    # Convert the list to a tensor type
    img = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(label_list)
    return img, y


def train_model(train_data):
    # Build the model
    network = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)])
    network.build(input_shape=(None, 100, 100, channels))
    network.summary()

    network.compile(optimizer=optimizers.SGD(lr=0.001),
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                    )
    checkpoint_filepath = "./model_save/gestureModel"
    callbacks = [
        # Callback function to save the model
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,  # Save path
                                           save_weights_only=True,
                                           verbose=0,
                                           save_freq='epoch'),  # Save frequency to calculate per epoch
        # Callback function to stop training
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)  # Prevent overfitting, stop training if the validation loss increases for 3 consecutive epochs
    ]
    # Just add callbacks=callbacks parameter in model.fit to save the model according to the designed callback function during training
    # Model training
    #network.load_weights('./model/gestureModel_one.h5')
    #print("Load trained weights successfully")
    network.fit(train_data, epochs=5, callbacks=callbacks)  # Since the dataset is a tuple with labels, there is no need to separate x and y
    # network.evaluate(test_data)
    network.save_weights('./model/gestureModel_one.h5')
    print("Save model weights successfully")
    tf.saved_model.save(network, './model/gestureModel_one')
    print("Save model successfully")
    return network


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, [100, 100])
    image /= 255.0  # normalize to [0,1] range
    # image = tf.reshape(image,[100*100*3])
    return image


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image), label

if __name__ == "__main__":

    # capture = cv.VideoCapture(0)
    # x1 = 400
    # x2 = 650
    # y1 = 50
    # y2 = 300
    # Path to training images
    global channels
    channels = 3
    train_dir = 'train_gesture_data'
    #test_dir = 'E:\\aiFile\\gesture_picture\\testdata'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Training images and labels
    image_list, label_list = get_files(train_dir)
    # Test images and labels
    #test_image_list, test_label_list = get_files(test_dir)

    # for i in range(len(image_list)):
    # print('Image path [{}] : Type [{}]'.format(image_list[i], label_list[i]))
    x_train, y_train = get_tensor(image_list, label_list)
    #x_test, y_test = get_tensor(test_image_list, test_label_list)

    # print('image_list:{}, label_list{}'.format(image_list, label_list))
    # print("train_image", image_list)
    # print("train_label", label_list)
    # print("img = ", img)
    # print("y = ", y)
    print('--------------------------------------------------------')
    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    # Generate CSV file for images and corresponding labels (only need to save once)
    # with open('./image_label.csv',mode='w', newline='') as f:
    #     Write = csv.writer(f)
    # for i in range(len(image_list)):
    #     Write.writerow([image_list[i], str(label_list[i])])
    # f.close()
    # Load training dataset
    db_train = tf.data.Dataset.from_tensor_slices((image_list, y_train))
    # # shuffle: shuffle the data, map: data preprocessing, batch: take 10 samples at a time for training
    db_train = db_train.shuffle(1000).map(load_and_preprocess_image).batch(10)
    #
    # # Load test dataset
    #db_test = tf.data.Dataset.from_tensor_slices((test_image_list, y_test))
    # # # shuffle: shuffle the data, map: data preprocessing, batch: take 10 samples at a time for training
    #db_test = db_test.shuffle(1000).map(load_and_preprocess_image).batch(10)
    print("dataset", db_train)
    network = train_model(db_train)
    # network = tf.keras.models.load_model('E:\\aiFile\\model_save\\model.h5')  # Load model
    # im = cv.imread("E:\\aiFile\\gesture_picture\\Dataset\\4\\101.jpg")
    # test_image = get_image(im)
    # test_pred = network.predict_classes(test_image)
    # print("Prediction = ", test_pred)
    # while True:
    #     ret, frame = capture.read()
    #     roi = get_roi(frame, x1, x2, y1, y2)
    #     cv.imshow("roi", roi)
    #     test_image = get_image(roi)
    #     test_pred = network.predict_classes(test_image)
    #     print("Prediction = ", test_pred[0])
    #     cv.imshow("frame", frame)
    #     c = cv.waitKey(50)
    #     if c == 27:
    #         break
    # cv.waitKey(0)
    # capture.release()
    # cv.destroyAllWindows()
