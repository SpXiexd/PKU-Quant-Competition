import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')


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


def test(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs,_ in test_loader:
            outputs = model(inputs.to(device))
            preds.append(outputs.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds


if __name__ == '__main__':
    net = torch.load("./net/Final 0.4934.pt")
    df = pd.DataFrame(columns=['stkid','group','pred_y001','pred_y002'])
    # 加载测试数据
    for i in range(8):
        fileName = "Testdata" + str(i + 1) + ".npz"
        test_data = np.load(fileName, allow_pickle=True)
        x_test = test_data["arr_0"]
        test_stock_id = test_data["arr_1"]
        missingStkid = test_data["arr_2"]
        x_test = torch.tensor(x_test, dtype=torch.float32)
        test_dataset = torch.utils.data.TensorDataset(x_test, x_test)
        test_loader = DataLoader(test_dataset, batch_size=1024)
        # 进行预测
        preds = test(net, test_loader)

        # 输出.csv文件
        groupID = np.ones(len(test_stock_id),dtype=np.int32)*(i+1)
        df2 = pd.DataFrame({'stkid':test_stock_id[:,1,1],'group':groupID,'pred_y001':preds[:,0],'pred_y002':preds[:,1]})
        nan_val = np.ones(len(missingStkid))*np.nan
        groupID2 = np.ones(len(missingStkid),dtype=np.int32)*(i+1)
        df3 = pd.DataFrame({'stkid':missingStkid,'group':groupID2,'pred_y001':nan_val,'pred_y002':nan_val})
        df = pd.concat([df, df2, df3], ignore_index=True)
    df.to_csv('preddata.csv', index=False)