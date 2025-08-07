import pywt
import numpy as np
from scipy import signal, stats
import pandas as pd

class WaveletFilter:
    def __init__(self, fs):
        self.fs = fs

    def apply(self, data):
        # Keep from 125 to 500 Hz
        lvl = int(np.log2(self.fs / 250))
        dw = pywt.wavedec(data, wavelet='db8', level=lvl)
        ndw = []
        for w in dw:
            ndw.append(np.zeros_like(w))
        ndw[1] = dw[1]
        ndw[2] = dw[2]

        fdata = pywt.waverec(ndw, wavelet='db8')

        return fdata


class WaveletEventDetector:
    def __init__(self, fs):
        self.fs = fs

    def apply(self, data):

        fdata = WaveletFilter(self.fs).apply(data)
        fdata = stats.zscore(fdata)
        henv = np.abs(signal.hilbert(fdata))**2
        shenv = pd.Series(henv).rolling(window=int(self.fs / 100), min_periods=1, center=True).mean()
        # thres = pd.Series(henv).rolling(window=int(self.fs), min_periods=1, center=True).mean()
        # msd = pd.Series(henv).rolling(window=int(self.fs), min_periods=1, center=True).std()
        # thres = thres + msd * 2
        thres = stats.iqr(shenv)
        peaksig = np.clip(shenv - thres, a_min=0, a_max=None)
        peaks, peak_prop = signal.find_peaks(peaksig, distance=1, height=0)

        return peaks, peaksig


if __name__ == '__main__':

    labels = train_file.get_labels()
    # train_file.show_data()
    events, peaksig = WaveletEventDetector(fs=fs).apply(data)

    train_file.show_data()
    plt.plot(xtime, peaksig[:-1])
    for e in events:
        plt.axvline(x=xtime[e], color='r')