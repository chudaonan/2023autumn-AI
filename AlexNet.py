import datetime
import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import cv2
import time

# 加载MNIST数据集并进行预处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
x_train /= 255.0
x_test /= 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# 定义AlexNet模型
def build_alexnet():
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(28, 28, 1), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())

    return model


# 创建AlexNet模型实例
alexnet_model = build_alexnet()

# 编译模型
alexnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 创建日志目录
log_dir = "C:/Users/lumic/Desktop/AI/keras_tald/logs/fit/an/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                      update_freq='epoch', profile_batch=0)

# 设置是否使用数据增强的变量，可以修改为True或False
use_data_augmentation = False

# 如果使用数据增强，定义一个数据生成器，设置不同的变换参数
if use_data_augmentation:
    datagen = ImageDataGenerator(
        rotation_range=10,  # 随机旋转角度范围
        width_shift_range=0.1,  # 随机水平平移范围
        height_shift_range=0.1,  # 随机垂直平移范围
        zoom_range=0.1,  # 随机缩放范围
        fill_mode='nearest'  # 填充模式
    )

# 定义训练参数
batch_size = 64
epochs = 10

# 如果使用数据增强，使用fit_generator方法来训练模型，传入数据生成器作为参数
# 否则，使用fit方法来训练模型，传入原始数据作为参数
if use_data_augmentation:
    start_time_alexnet = time.time()
    history_alexnet = alexnet_model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=len(x_train) // batch_size,
                                        epochs=epochs,
                                        validation_data=(x_test, y_test))
    end_time_alexnet = time.time()
else:
    # 训练AlexNet模型
    start_time_alexnet = time.time()
    history_alexnet = alexnet_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                        validation_data=(x_test, y_test))
    end_time_alexnet = time.time()

# 评估模型性能
_, acc_alexnet = alexnet_model.evaluate(x_test, y_test, verbose=0)
print("AlexNet模型准确率：", acc_alexnet)


# 测试画图板上的手写数字样本
def preprocess_image(image):
    # 调整图像大小为227x227
    image = cv2.resize(image, (227, 227))
    # 将图像转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # 将图像转换为浮点型并进行归一化
    image = image.astype("float32") / 255.0
    # 添加一个维度，以匹配模型的输入形状
    image = np.expand_dims(image, axis=0)
    return image


sample_images = []
num_images = 20

for i in range(1, num_images + 1):
    image_path = f"C:/Users/lumic/Desktop/AI/keras_tald/mnist/{i}.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法加载图像: {image_path}")
    else:
        sample_images.append(image)

num_rows = 4
num_cols = 5

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
fig.tight_layout()

for i, image in enumerate(sample_images):
    preprocessed_image = preprocess_image(image)
    row = i // num_cols
    col = i % num_cols

    axes[row, col].imshow(image)
    axes[row, col].axis("off")

    start_time_alexnet_prediction = time.time()
    prediction_alexnet = alexnet_model.predict(np.expand_dims(preprocessed_image, axis=0))
    end_time_alexnet_prediction = time.time()

    predicted_label_alexnet = np.argmax(prediction_alexnet)

    axes[row, col].set_title(f"alexnet: {predicted_label_alexnet}")

# 创建图像
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

# 保存图像
save_path = "C:/Users/lumic/Desktop/AI/keras_tald/mnist/an1.png"
plt.savefig(save_path)

plt.show()

print("alexnet模型推断时间：", end_time_alexnet_prediction - start_time_alexnet_prediction)
print("alexnet模型总运行时间：", end_time_alexnet - start_time_alexnet)
