import struct
import numpy as np
import os

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):      # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):     # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):       # 前向传播计算
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):       # 反向传播计算
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):         # 参数更新
        self.weight = self.weight - lr*self.d_weight
        self.bias = self.bias - lr*self.d_bias

    def load_param(self, weight, bias):     # 参数加载
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias       # 参数保存

class ReLULayer(object):
    def forward(self, input):       # 前向传播计算
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):       # 反向传播计算
        bottom_diff = top_diff
        bottom_diff[self.input<0] = 0
        return bottom_diff

class SoftmaxLossLayer(object):
    def forward(self, input):       # 前向传播计算
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):      # 损失函数计算
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):     # 反向传播计算
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=2,
                 print_iter=100, data_dir='', train_data='', train_label='', test_data='', test_label=''):
        # batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=2, print_iter=100
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.data_dir = data_dir
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

    def load_minst(self, file_dir, is_image = 'True'):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_image:    # 读取图像数据
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:           # 读取标记数据
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        train_images = self.load_minst(os.path.join(self.data_dir, self.train_data), True)
        train_labels = self.load_minst(os.path.join(self.data_dir, self.train_label), False)
        test_images = self.load_minst(os.path.join(self.data_dir, self.test_data), True)
        test_labels = self.load_minst(os.path.join(self.data_dir, self.test_label), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def build_model(self):
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def shuffle_data(self):       # 打乱训练过程数据
        np.random.shuffle(self.train_data)

    def train(self):
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                # self.update(self.lr)
                if idx_epoch < self.max_epoch / 4:
                    self.update(self.lr)
                elif idx_epoch < 3 * self.max_epoch / 4:
                    self.update(self.lr / (100 * ((3 * self.max_epoch / 4) - idx_epoch) / (self.max_epoch / 2)))
                else:
                    self.update(self.lr / 100)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    def evaluate(self):
        max_batch = int(self.test_data.shape[0] / self.batch_size)
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx_batch in range(max_batch):
            batch_images = self.test_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)

    def evaluateByLabel(self):
        max_batch = int(self.test_data.shape[0] / self.batch_size)
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx_batch in range(max_batch):
            batch_images = self.test_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size] = pred_labels
        each_label = np.zeros((self.out_classes, 2))
        for index, value in enumerate(pred_results):
            each_label[self.test_data[index, -1], 1] = each_label[self.test_data[index, -1], 1] + 1
            if pred_results[index] == self.test_data[index, -1]:
                each_label[self.test_data[index, -1], 0] = each_label[self.test_data[index, -1], 0] + 1
        accuracy = []
        for index in range(self.out_classes):
            accuracy.append(each_label[index, 0]/each_label[index, 1])
        return accuracy



# if __name__ == '__main__':
#     MNIST_DIR = "../Data/MINST_Train"
#     TRAIN_DATA = "train-images.idx3-ubyte"
#     TRAIN_LABEL = "train-labels.idx1-ubyte"
#     TEST_DATA = "t10k-images.idx3-ubyte"
#     TEST_LABEL = "t10k-labels.idx1-ubyte"
#     h1, h2, e = 20, 20, 1
#     mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e, data_dir=MNIST_DIR, train_data=TRAIN_DATA, train_label=TRAIN_LABEL, test_data=TEST_DATA, test_label=TEST_LABEL)
#     mlp.load_data()
#     mlp.build_model()
#     mlp.init_model()
#     mlp.train()
#     mlp.save_model('../Data/mlp-%d-%d-%depoch.npy' % (h1, h2, e))
#     mlp.evaluate()
#     mlp.load_model('../Data/mlp-%d-%d-%depoch.npy' % (h1, h2, e))
#     print(mlp.evaluateByLabel())


    # 训练过程学习率lr越小，需要越多的epoch进行迭代，可以考虑将lr动态设置，前期lr较大方便快速训练，后期lr逐渐变小方便收敛
    # 隐藏层结点个数越多，生成的函数会越近似于我们想要的函数，但是要放置结点数量过多导致的过拟合问题
