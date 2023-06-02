import numpy as np
from alive_progress import alive_bar
import time


def stack_arrays(input_array, output_array, time_step):
    num_samples, num_features, num_stkid = input_array.shape
    num_labels = output_array.shape[1]

    # 创建用于存储特征和标签的空数组
    # features = np.empty((0, time_step, num_samples - time_step, num_features))
    # labels = np.empty((0, num_labels, num_samples - time_step))
    features = np.empty(((num_samples - time_step)*num_stkid, time_step, num_features))
    labels = np.empty(((num_samples - time_step)*num_stkid, num_labels))
    step = num_samples - time_step
    # 对每个特征进行堆叠
    with alive_bar(num_stkid, force_tty=True) as bar:
        for i in range(num_stkid):
            temp1 = input_array[:, :, i]
            temp2 = output_array[:, :, i]

            # 使用高级索引和广播创建滑动窗口
            idx = np.arange(time_step)[:, None] + np.arange(num_samples - time_step)
            feature_windows = temp1[idx].transpose(1, 0, 2)
            label_windows = temp2[time_step:, :]

            # 将结果追加到数组中

            features[i*step:(i+1)*step,:,:] = feature_windows
            labels[i*step:(i+1)*step,:] = label_windows

            time.sleep(0.1)
            bar()

    return features, labels

if __name__ == '__main__':
    data = np.load("traindata_original.npz",allow_pickle=True)
    stkid = np.array(data['stkid']).reshape(-1,1)
    stkdate = np.array(data['date']).reshape(-1,1)
    keys = data.files
    fetureArry = np.array(data['x01']).reshape(-1,1)
    for i in range(3, len(keys)-2):
        temp = np.array(data[keys[i]]).reshape(-1,1)
        fetureArry = np.concatenate([fetureArry, temp], axis=1)
    y1 = np.array(data['y001']).reshape(-1,1)
    y2 = np.array(data['y002']).reshape(-1, 1)
    labelArry = np.concatenate([y1, y2], axis=1)
    stkid_unique = np.unique(stkid)
    stkdate_unique = np.unique(stkdate)

    # 初始化一个全零数组，shape为(len(stkdate_unique), len(fetureArry[0]), len(stkid_unique))
    input_array = np.zeros((len(stkdate_unique), len(fetureArry[0]), len(stkid_unique)))
    output_array = np.zeros((len(stkdate_unique), 2, len(stkid_unique)))
    # 获取stkid和stkdate的索引
    stkid_indices = {id: index for index, id in enumerate(stkid_unique)}
    stkdate_indices = {date: index for index, date in enumerate(stkdate_unique)}

    # 遍历fetureArry，将数据项放入正确的位置
    for i in range(len(stkid)):
        date_index = stkdate_indices[stkdate[i][0]]
        stkid_index = stkid_indices[stkid[i][0]]
        input_array[date_index, :, stkid_index] = fetureArry[i]
        output_array[date_index, :, stkid_index] = labelArry[i]

    # 用0替换nan值
    input_array = np.nan_to_num(input_array, nan=0.0)
    output_array = np.nan_to_num(output_array, nan=0.0)

    # 同个特征进行标准化
    for i in range(56):
        input_array[:,i,:] = (input_array[:,i,:] - np.nanmean(input_array[:,i,:])) / np.nanstd(input_array[:,i,:])
    for i in range(2):
        output_array[:, i, :] = (output_array[:, i, :] - np.nanmean(output_array[:, i, :])) / np.nanstd(output_array[:, i, :])

    time_step = 15
    np.random.seed(42)
    stock_num = 4116
    # 4116只股票太多了，分成3300个训练集和816个测试集合，其中训练集应该分成4个
    randomIndex = np.random.randint(0,input_array.shape[2],stock_num)
    Features1, Labels1 = stack_arrays(input_array[:,:,randomIndex[:825]], output_array[:,:,randomIndex[:825]], time_step)
    np.savez("trainData1.npz", Features1, Labels1)
    del Features1, Labels1
    Features2, Labels2 = stack_arrays(input_array[:, :, randomIndex[825:1650]], output_array[:, :, randomIndex[825:1650]],
                                    time_step)
    np.savez("trainData2.npz", Features2, Labels2)
    del Features2, Labels2
    Features3, Labels3 = stack_arrays(input_array[:, :, randomIndex[1650:2475]], output_array[:, :, randomIndex[1650:2475]],
                                    time_step)
    np.savez("trainData3.npz", Features3, Labels3)
    del Features3, Labels3
    Features4, Labels4 = stack_arrays(input_array[:, :, randomIndex[2475:3300]], output_array[:, :, randomIndex[2475:3300]],
                                    time_step)
    np.savez("trainData4.npz", Features4, Labels4)
    del Features4, Labels4
    Features5, Labels5 = stack_arrays(input_array[:, :, randomIndex[3300:4116]], output_array[:, :, randomIndex[3300:4116]],
                                    time_step)
    test_index = randomIndex[3300:4116]
    test_stock_id = np.empty((output_array.shape[0]-time_step)*816)
    for i in range(816):
        temp = np.ones(output_array.shape[0]-time_step)*test_index[i]
        test_stock_id[i*(output_array.shape[0]-time_step):(i+1)*(output_array.shape[0]-time_step)] = temp
    np.savez("testData1.npz", Features5, Labels5,test_stock_id)


