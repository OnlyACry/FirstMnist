import numpy as np
import NetWork
from tensorflow import keras

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

idx = np.random.choice(len(x_train), 10000, replace=False)  # 随机抽取1000个数据
x_train = x_train[idx]
y_train = y_train[idx]
idx = np.random.choice(len(x_test), 500, replace=False)
x_test = x_test[idx]
y_test = y_test[idx]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
#把每个图像转换成一个大小为 784 的一维数组，使其与神经网络输入的形状匹配。
x_train = x_train.reshape(len(x_train), 784, 1)
x_test = x_test.reshape(len(x_test), 784, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# print(y_test)
# print("\n")
# print(y_train)

if __name__ == "__main__":
    net = NetWork.Network([784, 30, 10])  # 输入层784 输出层10固定，线性层30
    training_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))
    net.SGD(training_data, 100, 100, 1, test_data)    # 参数依次为：训练数据集、训练周期、小批量数据大小、学习速率、测试数据集