# 1. 加载数据
from sklearn.datasets import load_diabetes
import torch
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target # 这里X和y是numpy数组，需要转换为torch的tensor张量才能传递到训练模型里
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1) # 这样可以把y显示转化为n行1列的二维张量，否则形状是[m,]的一维张量，在和模型输出对比的时候会出问题
X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]


# 2. 定义线性模型
from torch import nn
model = nn.Sequential(nn.Linear(10, 1))

# 3. 定义损失函数和优化器
loss = nn.MSELoss()
from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
from torch.utils import data
from sklearn.metrics import r2_score
def load_data(data_array, batch_size, is_train=True):
    """python数据迭代器，用来每次取一batch的数据"""
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 32
epoch_num = 3000

for epoch in range(epoch_num):
    losssum = 0.0
    for X_batch, y_batch in load_data((X_train, y_train), batch_size):
        l = loss(model(X_batch), y_batch) # 计算损失
        losssum += l.item()    # 累加损失 这里l本身是一个0维的tensor张量，用l.item()取出对应的python数值可以省空间
        l.backward() # 反向传播，计算梯度
        optimizer.step() # 更新参数
        optimizer.zero_grad() # 清空梯度
    if epoch % 10 == 0:
        l = losssum / (len(X_train) / batch_size)
        print(f'epoch: {epoch}, loss_train: {l:.4f}')

r2_train = r2_score(y_train, model(X_train).detach().numpy()) # 计算R2分数/这里的model已经是训练好的模型了
print(f'R2_train score max: {r2_train:.4f}')

# 5. 测试模型
l = loss(model(X_test), y_test) # 计算损失 losssum之前是因为训练时每个batch都计算了一次loss，所以要除以batch的个数，MSW本身对数据有自动取平均的功效，这里直接算一次loss就行
print(f'loss_test: {l:.4f}')

r2_test = r2_score(y_test, model(X_test).detach().numpy()) # 计算测试集上的R2分数
print(f'R2_test score: {r2_test:.4f}')

# Q1：没有用测试集检验模型 →所有的R2和loss都放到测试数据上计算 ✔
# Q2：每次R2的更新是用小的batch，随机性太大 →用整个训练集计算R2 ✔
# Q3：目前每个loss的计算都是只用一个batch，并没有用一个epoch的所有数据计算之后取平均值 →取平均值 ✔
# Q4：之前把lr甚至写成1这种夸张的值的时候r2还在升高，loss还在下降 → 数据比较温和，所以大步长也能收敛，但不推荐这么做；用0.01的时候r2小是因为步长小，epoch用100很难走到最优点上去，所以加大epoch数量时候就好了！