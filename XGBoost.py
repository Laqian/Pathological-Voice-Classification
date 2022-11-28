import matplotlib.pyplot as plt
import linecache
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor as XGBR
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# 从sklearn中加载测试方法
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

plt.style.use("ggplot")

voice = []
label = []

labels_map = {
    "healthy": 0,
    "hyperkinetic dysphonia": 1,
    "hypokinetic dysphonia": 2,
    "reflux laryngitis": 3,
}

voice_type = ['healthy', 'hyperkinetic dysphonia', 'hypokinetic dysphonia', 'reflux laryngitis']
voice_cnt = [0, 0, 0, 0]


def read_diagnosis(path):
    diagnosis = linecache.getline(path, 5)
    diagnosis_p = diagnosis[11:-1].strip().lower()
    for index in range(len(voice_type)):
        if voice_type[index] == diagnosis_p[:len(voice_type[index])]:
            return diagnosis_p[:len(voice_type[index])]
    return 0


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    # print '%s正确率：%.3f%%' % (tip, acc_rate)
    return acc_rate


if __name__ == '__main__':
    data_path = '../Data/VOICED/voice'
    info_path = '../Data/VOICED/label'
    root = '../Data/VOICED'
    files1 = os.listdir(data_path)
    files2 = os.listdir(info_path)
    for name in files2:
        label.append(labels_map[read_diagnosis(info_path + '/' + name)])
    for name in files1:
        voice_data = np.genfromtxt(data_path + '/' + name)
        # voice.append(voice_data[0:36000])
        # process_data = np.asarray(librosa.feature.melspectrogram(
        #     y=voice_data[1000:370000], sr=8000))
        # voice.append(process_data.reshape(process_data.shape[0] * process_data.shape[1])[0:9216])
        begin = 1000
        window = 8000
        step = 4000
        for time in range(7):
            # voice_spilit = librosa.feature.melspectrogram(
            #     y=voice_data[begin + time * step: begin + time * step + window],
            #     sr=8000)
            voice_spilit = voice_data[begin + time * step: begin + time * step + window]
            print(begin + time * step, begin + time * step + window)
            voice.append(voice_spilit)
    x = np.asarray(voice)
    y = np.asarray(label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    reg = XGBR(n_estimators=100).fit(x_train, y_train)  # 训练
    y_hat = reg.predict(x_test)  # 预测
    xgb_rate = show_accuracy(y_hat, y_test, 'XGB')
    print('XGBoost：%.3f%%' % xgb_rate)

    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_rate = show_accuracy(y_hat, y_test, 'Logistic回归 ')
    print('Logistic回归：%.3f%%' % lr_rate)
    #
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_rate = show_accuracy(y_hat, y_test, '随机森林 ')
    # write_result(rfc, 2)
    print('随机森林：%.3f%%' % rfc_rate)



    # clf = SVC(probability=True)
    # clf.fit(x_train, y_train)
    # # 进行交叉检验
    # kfold = KFold(n_splits=10)
    # result = cross_val_score(clf, x_train, y_train, cv=kfold)
    # print(result.mean())
