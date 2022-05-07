import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math
import os
import vad_utils as vad
import evaluate as eva

"""
已知音频文件采样率均为16kHz
帧长度32ms : frame_length = 512
帧移8ms : step = 128
"""


def div_seg(signal, frame_length=512, step=128):
    """
    分帧函数
    参数：
        signal 语音信号
        frame_length 帧长
        step 两帧之间的步长
    返回：
        seg 分割形成的二维数组
    """
    nx = len(signal)  # 语音信号的长度
    try:
        nwin = len(frame_length)
    except Exception as err:
        nwin = 1
    if nwin == 1:
        wlen = frame_length
    else:
        wlen = nwin
    nseg = int(np.fix((nx - wlen) / step) + 1)  # 窗口移动的次数
    seg = np.zeros((nseg, wlen))  # 初始化二维数组
    indf = [step * j for j in range(nseg)]
    indf = (np.mat(indf)).T
    inds = np.mat(range(wlen))
    indf_tile = np.tile(indf, wlen)
    inds_tile = np.tile(inds, (nseg, 1))
    mix_tile = indf_tile + inds_tile
    seg = np.zeros((nseg, wlen))
    for i in range(nseg):
        for j in range(wlen):
            seg[i, j] = signal[mix_tile[i, j]]
    return seg


def zero_cross_rate(signal, frame_length=512, step=128):
    """
    整段语音信号的过零率计算
    参数：
        signal 语音信号
        frame_length 帧长
        step 两帧之间的步长
    返回：
        zcr 过零率一维数组
    """

    L = len(signal)
    numOfFrames = np.asarray(np.ceil((L - frame_length) / step) + 1, dtype=int)
    zcr = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        curFrame = signal[np.arange(i * step, min(i * step + frame_length, L))]
        curFrame = curFrame - np.mean(curFrame)  # zero-justified
        zcr[i] = sum(curFrame[0:-1] * curFrame[1::] <= 0)
    return zcr


def short_time_energy(signal, frame_length=512, step=128):
    """
    整段语音信号的每帧短时能量
    参数：
        signal 语音信号
        windowLength 帧长
        step 帧移
    返回：
        energy 每帧能量一维数组
    """
    # signal = signal / np.max(signal)  # 归一化
    curPos = 0
    L = len(signal)
    numOfFrames = np.asarray(np.ceil((L - frame_length) / step) + 1, dtype=int)
    energy = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        window = signal[int(curPos) : int(curPos + frame_length - 1)]
        energy[i] = (1 / (frame_length)) * np.sum(np.abs(window**2))
        curPos = curPos + step

    # 能量归一化
    energy = energy / np.max(energy)
    return energy


def sort_by_thres(threshold, key):
    if threshold - key > 0:
        return 0
    else:
        return 1


def get_acc(prediction, actual):
    """
    input:
        prediction 预测list eg.[0,0,1,0,...,1,0]
        actual 从实际label文件中读取的list
    return:
        acc accuracy
    """
    total_frame = len(actual)
    correct_frame = 0

    for i in range(total_frame):
        if prediction[i] == actual[i]:
            correct_frame += 1
    acc = correct_frame / total_frame

    # voice_frame = 0
    # correct_voice_frame = 0
    # for i in range(total_frame):
    #     if actual[i] == 1:
    #         voice_frame += 1
    #         if prediction[i] == actual[i]:
    #             correct_frame += 1
    # tpr = correct_voice_frame / voice_frame

    # nonvoice_frame = 0
    # false_nonvoice_frame = 0
    # for i in range(total_frame):
    #     if actual[i] == 0:
    #         nonvoice_frame += 1
    #         if prediction[i] != actual[i]:
    #             false_nonvoice_frame += 1
    # fpr = false_nonvoice_frame / nonvoice_frame

    return acc


# 路径
wav_path = "../vad/wavs/dev"
label_path = "../vad/data"

# 读label文件
label_data = vad.read_label_from_file(label_path + "/dev_label.txt")

# 读取文件夹
files = os.listdir(wav_path)

result_energy = []
label = []

print("\nProcessing dev data...\n")
for file in files:
    # print("Processing " + wav_path + "/" + file)
    # 读取音频文件
    sample_rate, data = wavfile.read(wav_path + "/" + file)

    # 分帧计算短时能量
    energy = short_time_energy(data, 512, 128)

    # 计算帧数
    L = len(data)
    num_of_steps = np.asarray(np.ceil((L - 512) / 128) + 1, dtype=int)

    # 时间轴
    time = np.zeros(num_of_steps)
    for i in range(num_of_steps):
        time[i] = (i * 128 + 256) / 16000

    # 补零
    current_label_data = label_data[file[0:-4]]
    current_label_data = list(current_label_data) + list(
        np.zeros(len(time) - len(current_label_data))
    )

    for i in range(len(current_label_data)):
        label.append(current_label_data[i])

    # for i in range(num_of_steps):
    #     result.append(sort_by_thres(0.648, energy[i]))

    for i in range(num_of_steps):
        result_energy.append(energy[i])

# vad.parse_vad_label(label)
# print(len(result))
# print(len(vad.parse_vad_label(label)))
# print(len(label))
print("\nCalculating AUC, EER, TPR, FPR, Threshold of dev dataset...\n")
auc, eer, tpr, fpr, thres = eva.get_metrics(result_energy, label)
print("AUC = ", auc)
print("EER = ", eer)
print("TPR = ", tpr)
print("FPR = ", fpr)
print("Threshold = ", thres)

result2_energy = []
for i in range(len(result_energy)):
    result2_energy.append(sort_by_thres(thres, result_energy[i]))

print("\nCalculating ACC...\n")
acc = get_acc(result2_energy, label)
print("ACC = ", acc)


plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.savefig("task1_ROC_dev.png")
plt.show()


"""
------------
进入测试集
------------
"""
print("\nProcessing test data...\n")
# 读取文件夹
wav_path = "../vad/wavs/test"
files = os.listdir(wav_path)
f = open("test.txt", "w")

for file in files:
    # print("Processing " + wav_path + "/" + file)
    # 读取音频文件
    sample_rate, data = wavfile.read(wav_path + "/" + file)

    # 分帧计算短时能量
    energy = short_time_energy(data, 512, 128)

    # 计算帧数
    L = len(data)
    num_of_steps = np.asarray(np.ceil((L - 512) / 128) + 1, dtype=int)

    # 时间轴
    time = np.zeros(num_of_steps)
    for i in range(num_of_steps):
        time[i] = (i * 128 + 256) / 16000

    # for i in range(num_of_steps):
    #     result.append(sort_by_thres(0.648, energy[i]))
    test_result = []
    for i in range(num_of_steps):
        # test_result.append(energy[i])
        test_result.append(sort_by_thres(thres, energy[i]))

    test_label = []
    test_label = vad.prediction_to_vad_label(test_result)

    # 写入txt
    f.write(file[0:-4] + " " + test_label + "\n")

f.close()

print("\nComplete!\nResult file generated as test.txt\n")

# """
# 短时能量
# 使用54-121080-0009.wav文件进行测试
# 发现阈值应在0.3左右
# """
# # 读取特定音频文件
# sample_rate, data = wavfile.read(wav_path + "/54-121080-0009.wav")

# # 显示采样率
# print("Sample rate =", sample_rate)

# # 分帧计算短时能量
# energy = short_time_energy(data, 512, 128)

# # 计算帧数
# L = len(data)
# num_of_steps = np.asarray(np.ceil((L - 512) / 128) + 1, dtype=int)

# # 时间轴
# time = np.zeros(num_of_steps)
# for i in range(num_of_steps):
#     time[i] = (i * 128 + 256) / 16000

# # # 绘制短时能量
# # plt.plot(time, energy)

# # 读取当前音频文件的label
# current_label_data = label_data["54-121080-0009"]

# # 补零
# current_label_data = list(current_label_data) + list(
#     np.zeros(len(time) - len(current_label_data))
# )
# for i in range(len(current_label_data)):
#     label.append(current_label_data[i])
# for i in range(num_of_steps):
#     result.append(sort_by_thres(0.3, energy[i]))

# print(label)
# print("+++++++++++++++")
# print(result)
# print("+++++++++++++++")
# print(len(label), len(result))

# vad.parse_vad_label(label)
# print(len(result))
# print(len(vad.parse_vad_label(label)))
# print(len(label))
# auc, eer, thres = eva.get_metrics(result, label)
# print(auc, eer, thres)


# # 绘制label
# plt.plot(time, current_label_data)
# plt.show()
