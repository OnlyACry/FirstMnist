import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes  ##神经网络层数
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] ##对网络中除了第一层(输入层)的神经元设置随机偏置量
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # random.shuffle(training_data)   #打乱顺序
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #训练一小组

            if test_data != None:
                num = self.evaluate(test_data)
                print("周期:{0}  {1}/{2} {3}%".format(j + 1, num, n_test, num / n_test * 100))
            else:
                print("周期 {0}  完成！".format(j + 1))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #计算局部最优解
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)    #加入list
            activation = sigmoid(z)
            activations.append(activation)
        #计算输出层误差
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # delta = delta.reshape((10, 1))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(x == y) for x, y in test_result)

    def cost_derivative(self, output_activations, y):
        # y = np.reshape(y, output_activations.shape) #尝试解决y与output_activations不兼容
        # print(output_activations)
        # print(y)

        # return (output_activations-y)

        output_activations = output_activations[:]
        index = np.argmax(y)    #找到预期输出
        output_activations[index] -= 1
        return output_activations

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

'''计算sigmoid导数'''
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))