import numpy as np
import pandas as pd
from alive_progress import alive_bar
import time

def stack_arrays(input_array, stkid_array, time_step):
    num_samples, num_features, num_stkid = input_array.shape
    # 创建用于存储特征和标签的空数组
    # features = np.empty((0, time_step, num_samples - time_step, num_features))
    features = np.empty((num_stkid, time_step, num_features))
    outSTK = np.empty((num_stkid, time_step, num_features),  dtype='U20')

    step = num_samples - time_step
    # 对每个特征进行堆叠
    with alive_bar(num_stkid, force_tty=True) as bar:
        for i in range(num_stkid):
            temp1 = input_array[:, :, i]
            temp2 = stkid_array[:, :, i]
            # 使用高级索引和广播创建滑动窗口
            idx = np.arange(time_step)[:, None] + np.arange(num_samples - time_step)
            idx2 = idx[:,39].reshape(10,1)  # 只取最后一天
            feature_windows = temp1[idx2].transpose(1, 0, 2)
            stkd_windows = temp2[idx2].transpose(1, 0, 2)
            # 将结果追加到数组中
            features[i,:,:] = feature_windows
            outSTK[i, :, :] = stkd_windows
            time.sleep(0.1)
            bar()
    return features, outSTK


if __name__ == "__main__":
    data = np.load("testdata.npz", allow_pickle=True)
    stkid = np.array(data['stkid']).reshape(-1, 1)
    groupid = np.array(data['group']).reshape(-1, 1)
    date = np.array(data['dayOrder']).reshape(-1, 1)

    # 提取出day=50的stkid
    labeldf = pd.DataFrame({**data})[['stkid', 'dayOrder', 'group']]
    labeldf = labeldf.loc[labeldf.dayOrder == 50].copy()


    keys = data.files
    fetureArry = np.array(data['x01']).reshape(-1, 1)
    for i in range(4, len(keys)):
        temp = np.array(data[keys[i]]).reshape(-1, 1)
        fetureArry = np.concatenate([fetureArry, temp], axis=1)
    stkid_unique = np.unique(stkid)
    groupid_unique = np.unique(groupid)
    date_unique = np.unique(date)

    input_array = np.zeros((len(groupid_unique), len(date_unique), len(fetureArry[0]), len(stkid_unique)))
    stkid_array = np.empty((len(groupid_unique), len(date_unique), len(fetureArry[0]), len(stkid_unique)),  dtype='U20')
    # 获取stkid和stkdate的索引
    stkid_indices = {id: index for index, id in enumerate(stkid_unique)}
    stkdate_indices = {date: index for index, date in enumerate(date_unique)}
    groupid_indices = {group: index for index, group in enumerate(groupid_unique)}

    # 遍历fetureArry，将数据项放入正确的位置
    for i in range(len(stkid)):
        date_index = stkdate_indices[date[i][0]]
        stkid_index = stkid_indices[stkid[i][0]]
        group_index = groupid_indices[groupid[i][0]]
        input_array[group_index, date_index, :, stkid_index] = fetureArry[i]
        stkid_array[group_index, date_index, :, stkid_index] = stkid[i][0]

    # 用0替换nan值
    input_array = np.nan_to_num(input_array, nan=0.0)
    # 同个特征进行标准化
    for i in range(56):
        input_array[:, :, i, :] = (input_array[:, :, i, :] - np.nanmean(input_array[:, :, i, :])) / np.nanstd(
            input_array[:, :, i, :])

    time_step = 10
    for k in [0,1,2,4,5,6]:
        fileName = "Testdata" + str(k+1) + ".npz"
        day50 = labeldf.loc[labeldf.group == k+1]['stkid'].to_numpy()
        Features, outSTK = stack_arrays(input_array[k, :, :, :], stkid_array[k, :, :, :], time_step)
        stkid_vector = outSTK[:,1,1]
        intersection = np.intersect1d(stkid_vector, day50, return_indices=True)  # 一维是交集，二维是交集在第一个数组中的索引，三维是交集在第二个数组的索引
        # missing_indices = np.where(np.diff(intersection[2]) > 1)[0]  # 找到stkid_vector中缺少的数值
        # missingSTK1 = day50[missing_indices + 1]
        Features2 = Features[intersection[1], :, :]
        outSTK2 = outSTK[intersection[1], :, :]
        missingSTK1 = np.setdiff1d(day50, intersection[0])
        np.savez(fileName, Features2, outSTK2, missingSTK1)

