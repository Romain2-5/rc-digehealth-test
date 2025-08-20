from sklearn.metrics import classification_report
from src.rc_digehealth_test.utils import DigeHealthFile, find_nonzero_segments, fill_label_gaps
from src.rc_digehealth_test.preprocessors import ButterFilter
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb


def get_x_y(file):

    fs = file.fs


    _, seg_labels = file.get_segmented_data(segment_size_sec=0.05, group_bursts=True)
    lab = np.array([int(np.unique(l)[0]) for l in seg_labels])

    bfilter = ButterFilter(cutoff=80, order=2, fs=fs, btype='highpass')
    xtime, data = file.get_raw_data()
    data = bfilter.apply(data)
    win = signal.windows.hamming(int(0.05*fs))
    sft = signal.ShortTimeFFT(win=win, hop=int(0.05*fs), fs=fs)

    sx = np.abs(sft.stft(data))[:, :len(lab)]

    sx = 10 * np.log10(sx)
    sx = sx / np.sum(sx, axis=0, keepdims=True)

    return sx.T, lab


if __name__ == '__main__':
    # Get the files
    train_file = DigeHealthFile('../test-data/AS_1.wav', decimate_data=True)
    test_file = DigeHealthFile('../test-data/23M74M.wav', decimate_data=True)

    xtrain, ytrain = get_x_y(train_file)
    xtest, ytest = get_x_y(test_file)

    freq = np.linspace(0, train_file.fs/2, xtrain.shape[1])

    plt.plot(freq, np.mean(xtrain[ytrain==0], 0), label='No noise')
    plt.plot(freq, np.mean(xtrain[ytrain==1], 0), label='One burst')
    plt.plot(freq, np.mean(xtrain[ytrain==3], 0), label='Multi burst')
    plt.plot(freq, np.mean(xtrain[ytrain==2], 0), label='Harmonic')
    plt.legend()
    plt.ylabel('Power (Db)')
    plt.xlabel('Frequency (Hz)')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.show()

    gbm = lgb.LGBMClassifier(class_weight='balanced')
    gbm.fit(xtrain, ytrain)

    pred = gbm.predict(xtest)
    print(classification_report(ytest, pred))
