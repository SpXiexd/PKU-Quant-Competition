import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from memory_profiler import profile
import time
import torch.cuda.amp as amp
import pandas as pd
import matplotlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')

# 定义 LSTM 网络
class StockLSTM(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super(StockLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)  # 初始化LSTM参数
        self.fc = nn.Linear(hidden_size, output_size)
        self.apply(self.weight_init)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 隐藏状态，可能跟上面手动初始化冲突
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 细胞状态
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 定义训练函数
@profile(precision=4, stream=open('test.log', 'w+'))
def train(model, optimizer, criterion, train_loader_set, valid_loader_set, num_epochs, scaler, autocast, scheduler):
    for epoch in range(num_epochs):
        model.train()
        global train_loss_list, vail_loss_list, corelation_list
        train_loss = 0.0
        train_step = 0
        for batch, data in enumerate(zip(train_loader_set[0],train_loader_set[1],train_loader_set[2],train_loader_set[3])):
            for i in range(4):
                inputs, targets = data[i]
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs = inputs.float()
                targets = targets.float()

                # outputs = model(inputs)
                # loss = criterion(outputs, targets).sum()
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # 使用 autocast 控制计算精度
                # 混合精度AMP可能可以加快训练速度
                optimizer.zero_grad()
                with autocast:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets).sum()
                # 使用 GradScaler 进行梯度缩放
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                train_loss += loss.item()
                train_step += 1
        train_loss /= train_step
        train_loss_list.append(train_loss)

        #　验证集
        model.eval()
        valid_loss = 0.0
        preds = []
        targets = []
        vali_step = 0
        with torch.no_grad():
            for batch, valiData in enumerate(zip(valid_loader_set[0], valid_loader_set[1], valid_loader_set[2], valid_loader_set[3])):
                for i in range(4):
                    inputs, labels = valiData[i]
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    inputs = inputs.float()
                    labels = labels.float()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels).sum()
                    valid_loss += loss.item()
                    vali_step += 1

                    # preds.append(outputs.cpu().numpy())
                    # targets.append(labels.cpu().numpy())

                    # detach()避免将数据从GPU移到CPU上的开销
                    preds.append(np.asarray(outputs.detach().cpu()))
                    targets.append(np.asarray(labels.detach().cpu()))
        valid_loss /= vali_step
        vail_loss_list.append(valid_loss)

        # 求斯皮尔曼秩相关系数的平均值
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        corr = np.mean([spearmanr(preds[:, i], targets[:, i]).correlation for i in range(targets.shape[1])])
        corelation_list.append(corr)
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
             f"Valid Corr: {corr:.4f}, Train Time:{end_time-start_time:4f}")
    # train loss，vali loss，spearmanr corelation
    fig,ax = plt.subplots(2,1,figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)
    ax1 = ax[0]
    ax2 = ax[1]
    ax1.plot(range(1, num_epochs + 1), train_loss_list, linestyle='-', label="train")
    ax1.plot(range(1, num_epochs + 1), vail_loss_list, linestyle=':', label="validation")
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Train and Validation Loss')

    ax2.plot(range(1, num_epochs + 1), corelation_list, linestyle=':', label="corelation")
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Corelation', fontsize=14)
    ax2.set_title('Spearmanr Corelation')
    plt.savefig('./Correlation2.png')
    plt.legend()
    plt.show()


# 定义测试函数
def test(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs,_ in test_loader:
            outputs = model(inputs.to(device))
            preds.append(outputs.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds


def saveCsv(stockID,pred,groupID):
    df = pd.DataFrame({'stkid':stockID,'group':groupID,'pred_001':pred[:,0],'pred_002':pred[:,1]})
    df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    # 定义损失函数和优化器
    train_loss_list, vail_loss_list, corelation_list = [], [], []
    Epochs = 50
    batch_size = 1024
    num_workers = 0
    scaler = amp.GradScaler()
    autocast = amp.autocast()
    criterion = nn.MSELoss().to(device)
    # 定义模型
    model = StockLSTM(input_size=56, hidden_size=512, output_size=2, num_layers=6, dropout=0).to(device)
    model = model.float()  # 解决代码中网络参数类型不统一
    learning_rate = 0.002
    # 定义 GradScaler 和 autocast
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)  # 效果看起来比Adam优化器收敛快
    # 设置震荡学习率
    # 定义 CyclicLR 调度器
    # time_step=5，batch_size=1024时，一个训练集的迭代数为845*1764/1024约1421，此时的step_size可以设置为2到10倍，即是2842或14210
    clr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=0.006, step_size_up=8000,
    step_size_down=8000, mode='triangular2', cycle_momentum=False)
    train_loader_set = []
    val_loader_set = []

    #  读取四个训练集
    for i in range(4):  # 4
        start_time = time.time()
        filename = "trainData"+str(i+1) + ".npz"
        data = np.load(filename,allow_pickle=True)
        Fetures = data["arr_0"]
        Labels = data["arr_1"]
        # random_state是随机种子
        X_train, X_val, Y_train, Y_val = train_test_split(Fetures, Labels, test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32)
        # 只有将数据转化成Tensor才能使用torch.utils.data.TensorDataset，同时自定义Dataset无法用多线程load data
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
        val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True, drop_last=False)
        train_loader_set.append(train_loader)
        val_loader_set.append(val_loader)
        del val_loader,train_loader,X_train, X_val, Y_train, Y_val
    train(model, optimizer, criterion, train_loader_set, val_loader_set, Epochs, scaler, autocast, clr_scheduler)

    del train_loader_set, val_loader_set
    # 加载测试数据
    test_data = np.load("testData1.npz")
    x_test = test_data["arr_0"]
    true_data = test_data["arr_1"]
    test_stock_id = test_data["arr_2"]
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(true_data, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024)
    # 进行预测
    preds = test(model, test_loader)
    spearmanArry = [spearmanr(preds[:, i], true_data[:, i]).correlation for i in range(true_data.shape[1])]
    corr = np.mean(spearmanArry)
    print(f"Average Spearman correlation coefficient: {corr:.4f}")

    # 保存预测结果
    np.savetxt("predictions.csv", preds, delimiter=",")
    torch.save(model,"test1.pt")

    # 输出.csv文件
    groupID = np.ones(len(test_stock_id))
    saveCsv(test_stock_id, preds, groupID)