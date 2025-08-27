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
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 训练模型
from torch.utils import data
from sklearn.metrics import r2_score
def load_data(data_array, batch_size, is_train=True):
    """python数据迭代器，用来每次取一batch的数据"""
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 32
epoch_num = 100
r2max = -1
for epoch in range(epoch_num):
    for X_batch, y_batch in load_data((X_train, y_train), batch_size):
        l = loss(model(X_batch), y_batch) # 计算损失
        r2 = r2_score(y_batch, model(X_batch).detach().numpy()) # 计算R2分数
        if r2 > r2max: # 只在后半段保存模型
            r2max = r2
            torch.save(model.state_dict(), 'linear.pth') # 保存模型 后面可以用相同模型来承接
        l.backward() # 反向传播，计算梯度
        optimizer.step() # 更新参数
        optimizer.zero_grad() # 清空梯度
    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {l:.4f}')
        print(f'R2 score max: {r2max:.4f}')
        
