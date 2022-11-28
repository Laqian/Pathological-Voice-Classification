import linecache
import os
import numpy
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display

voice = []
label = []

voice_type = ['healthy', 'hyperkinetic dysphonia', 'hypokinetic dysphonia', 'reflux laryngitis']
voice_cnt = [0, 0, 0, 0]


def read_diagnosis(path):
    diagnosis = linecache.getline(path, 5)
    diagnosis_p = diagnosis[11:-1].strip().lower()
    for index in range(len(voice_type)):
        if voice_type[index] == diagnosis_p[:len(voice_type[index])]:
            return diagnosis_p[:len(voice_type[index])]
    return 0


if __name__ == '__main__':
    data_path = '../Data/VOICED/voice'
    info_path = '../Data/VOICED/label'
    root = '../Data/VOICED'
    files1 = os.listdir(data_path)
    files2 = os.listdir(info_path)
    for name in files2:
        label.append(read_diagnosis(info_path + '/' + name))
    cnt = 0
    for name in files1:
        voice_data = np.genfromtxt(data_path + '/' + name)
        # if cnt == 9:
        #     melspec = librosa.feature.melspectrogram(y=voice_data[1000:7000], sr=8000)
        #     logmelspec = librosa.power_to_db(melspec)  # 转换为对数刻度
        #     # 绘制 mel 频谱图
        #     plt.figure()
        #     librosa.display.specshow(logmelspec, sr=8000, x_axis='time', y_axis='mel')
        #     plt.colorbar(format='%+2.0f dB')  # 右边的色度条
        #     plt.title('')
        #     plt.show()

        begin = 1000
        window = 6000
        step = 1000
        # numpy.savetxt(root + '/mfcc/' + label[cnt] + '.' +
        #               str(voice_cnt[voice_type.index(label[cnt])]) + '.txt',
        #               librosa.feature.melspectrogram(y=voice_data, sr=8000))
        # voice_cnt[voice_type.index(label[cnt])] += 1
        # cnt += 1
        for time in range(31):
            # print(librosa.feature.melspectrogram(y=voice_data, sr=8000).shape)
            # print(begin + time * step, begin + time * step + window)
            # voice_split = librosa.feature.melspectrogram(
            #     y=voice_data[begin + time * step: begin + time * step + window],
            #     sr=8000)

            voice_split = voice_data[begin + time * step: begin + time * step + window]
            # print(len(voice_split))
            # print(cnt, label[cnt], voice_cnt[voice_type.index(label[cnt])])
            numpy.savetxt(root + '/data/' + label[cnt] + '.' +
                          str(voice_cnt[voice_type.index(label[cnt])]) + '.txt',
                          voice_split)
            voice_cnt[voice_type.index(label[cnt])] += 1
        cnt += 1

    # count the type
    # for name in label:
    #     for index in range(len(voice_type)):
    #         if voice_type[index] == name:
    #             voice_cnt[index] += 1
    # print(voice_cnt)
