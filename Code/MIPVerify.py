import gurobipy
import numpy as np
import MINST_Train

MNIST_DIR = "../Data/MINST_Train"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"


class MIPVerify(object):
    def __init__(self, nnmodel, index, epsilon, target_index):
        self.nnmodel = nnmodel
        self.index = index
        self.epsilon = epsilon
        self.target_index = target_index
        self.mipmodel = gurobipy.Model()

    def addReLUConstr(self, lower, upper, x):    # relu() ==> y = max(0,x)
        if upper < 0:
            return 0
        elif lower > 0:
            return x
        else:
            y = self.mipmodel.addVar(lb=0, ub=upper, vtype=gurobipy.GRB.CONTINUOUS)
            a = self.mipmodel.addVar(vtype=gurobipy.GRB.BINARY)
            self.mipmodel.addConstr(y <= x - lower * (1 - a))
            self.mipmodel.addConstr(y >= x)
            self.mipmodel.addConstr(y <= upper * a)
            self.mipmodel.addConstr(y >= 0)
            return y

    def bound_compute(self, bound, weight, bias):
        b = np.zeros((weight.shape[1], 2))
        for wIndex, wValue in enumerate(weight):
            for bIndex, bValue in enumerate(wValue):
                if bValue < 0:
                    b[bIndex, 0] = b[bIndex, 0] + bValue * bound[wIndex, 1]
                    b[bIndex, 1] = b[bIndex, 1] + bValue * bound[wIndex, 0]
                else:
                    b[bIndex, 0] = b[bIndex, 0] + bValue * bound[wIndex, 0]
                    b[bIndex, 1] = b[bIndex, 1] + bValue * bound[wIndex, 1]
        return b+bias.T

    def load_model(self, h1, h2, e):
        self.nnmodel.load_data()
        self.nnmodel.build_model()
        self.nnmodel.init_model()
        self.nnmodel.load_model('../Data/mlp-%d-%d-%depoch.npy' % (h1, h2, e))
        self.picture = self.nnmodel.test_data[self.index][0:-1]
        self.label = self.nnmodel.test_data[self.index][-1]

    def addPreconditionConstr(self):
        perturbation_added = []
        input_with_PA = []
        for index in range(self.nnmodel.input_size):
            perturbation_added.append(self.mipmodel.addVar(lb=-self.epsilon, ub=self.epsilon, vtype=gurobipy.GRB.CONTINUOUS))
            input_with_PA.append(self.mipmodel.addVar(lb=max(0, self.picture[index] - self.epsilon), ub=min(255, self.picture[index] + self.epsilon), vtype=gurobipy.GRB.CONTINUOUS))
            self.mipmodel.addConstr(input_with_PA[-1] == self.picture.tolist()[index] + perturbation_added[-1])
        return input_with_PA, perturbation_added

    def addNNConstr(self, preVar, postVarCount, weight, bias, bound):
        temp = []
        np.maximum(bound, 0)
        postVar = []
        b = self.bound_compute(bound, weight, bias)  # bound = [lower, upper]
        for i in range(postVarCount):
            expr = gurobipy.LinExpr()
            for j in range(len(preVar)):
                if isinstance(preVar[j], int):
                    expr.addConstant(0)
                else:
                    expr.addTerms(weight.tolist()[j][i], preVar[j])
            expr.addConstant(bias.tolist()[0][i])
            temp.append(expr)
            postVar.append(self.addReLUConstr(b[i, 0], b[i, 1], expr))
        return postVar, b, temp

    def addPostconditionConstr(self, preVar, postVarCount, weight, bias):
        postVar = []
        for i in range(postVarCount):
            expr = gurobipy.LinExpr()
            for j in range(len(preVar)):
                if isinstance(preVar[j], int):
                    expr.addConstant(0)
                else:
                    expr.addTerms(weight.tolist()[j][i], preVar[j])
            expr.addConstant(bias.tolist()[0][i])
            postVar.append(expr)
        # 设定目标函数
        for i in range(10):
            if i != self.target_index:
                self.mipmodel.addConstr(postVar[i] <= postVar[self.target_index])
        # model.setObjective(postVar[7] - postVar[0], gurobipy.GRB.MINIMIZE)

    def setObj(self, perturbation_added):
        obj = self.mipmodel.addVar(vtype=gurobipy.GRB.CONTINUOUS)
        for index, value in enumerate(perturbation_added):
            x_abs = self.mipmodel.addVar(lb=0.0, ub=self.epsilon, vtype=gurobipy.GRB.CONTINUOUS)
            self.mipmodel.addConstr(x_abs >= perturbation_added[index])
            self.mipmodel.addConstr(x_abs >= -1*perturbation_added[index])
            self.mipmodel.addConstr(obj >= x_abs)
        self.mipmodel.setObjective(obj, gurobipy.GRB.MINIMIZE)


# if __name__ == '__main__':
#     # 读取训练好的模型
#     h1, h2, e = 32, 16, 1
#     mlp = MINST_Train.MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e, data_dir=MNIST_DIR, train_data=TRAIN_DATA,
#                                 train_label=TRAIN_LABEL, test_data=TEST_DATA, test_label=TEST_LABEL)
#
#     # 选取一张图片
#     index = 1   # 图片序号
#     epsilon = 50   # 扰动范围
#     picture = mlp.test_data[index][0:-1]
#     label = mlp.test_data[index][-1]    # label = 7
#     target_index = 9
#     print(label)
#
#     # 创建MIP模型,创建变量并更新变量空间
#     model = MIPVerify(nnmodel=mlp, index=index, epsilon=epsilon, target_index=target_index)
#     model.load_model(h1, h2, e)
#     input_with_PA, perturbation_added = model.addPreconditionConstr()
#
#     bound = np.zeros((mlp.input_size, 2))
#     bound[:, 0] = mlp.test_data[index][0:-1] - epsilon
#     bound[:, 1] = mlp.test_data[index][0:-1] + epsilon
#     np.minimum(bound, 255)
#     np.maximum(bound, 0)
#     hidden1, bound, temp1 = model.addNNConstr(input_with_PA, mlp.hidden1, mlp.fc1.weight, mlp.fc1.bias, bound)
#     hidden2, bound, temp2 = model.addNNConstr(hidden1, mlp.hidden2, mlp.fc2.weight, mlp.fc2.bias, bound)
#     model.addPostconditionConstr(hidden2, mlp.out_classes, mlp.fc3.weight, mlp.fc3.bias)
#     model.setObj(perturbation_added)
#
#     # 执行最优化过程
#     model.mipmodel.optimize()
#     # model.computeIIS()
#     # model.write("model1.ilp")
#     model.mipmodel.write("model1.lp")
#     print("Obj:", model.mipmodel.objVal)