from scipy.io import wavfile
from scipy.signal import decimate
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats


class DigeHealthFile:

    def __init__(self, digehealth_file_path, decimate_data=False):
        self.digehealth_file_path = digehealth_file_path
        original_fs, self.raw_data = wavfile.read(digehealth_file_path)

        if decimate_data:
            self.raw_data = decimate(self.raw_data, int(original_fs / 16000))
            self.fs = 16000
        else:
            self.fs = original_fs

        self.xtime = np.arange(0, len(self.raw_data) / self.fs, 1 / self.fs)

        if os.path.exists(digehealth_file_path.replace('.wav', '.txt')):
            self.labels = pd.read_csv(digehealth_file_path.replace('.wav', '.txt'), header=0, sep='\t',
                                    names=['start', 'end', 'label'])
            self.labels.loc[self.labels.label == 'sb', 'label'] = 'b' # it's not consistent across files.
        else:
            self.labels = None

    def get_raw_data(self):
        return self.xtime, self.raw_data

    def get_labels(self):
        return self.labels

    def show_data(self, preprocessor=None):
        xtime, data = self.get_raw_data()

        if preprocessor is not None:
            data = preprocessor.apply(data)

        fig, ax = plt.subplots()
        ax.plot(xtime, data)
        for i in range(len(self.labels)):
            match self.labels.iloc[i]['label']:
                case 'h':
                    color = 'tab:red'
                case 'b':
                    color = 'tab:blue'
                case 'mb':
                    color = 'tab:green'
                case _:
                    color = None
            if color:
                ax.axvspan(self.labels.iloc[i]['start'], self.labels.iloc[i]['end'], color=color, alpha=0.2)
        plt.show()

    def get_segmented_data(self, segment_size_sec=0.6, filter_to_apply=None, normalize=True):
        label_map = {'b': 1, 'mb': 2, 'h': 3}
        self.labels['label_num'] = self.labels['label'].map(label_map).fillna(0).astype(int)

        xtime_lookup = np.searchsorted(self.xtime, self.labels[['start', 'end']].values, side='left')
        label_num = np.zeros_like(self.raw_data)

        for idx, (start_idx, end_idx) in enumerate(xtime_lookup):
            label_value = self.labels['label_num'].iloc[idx]
            label_num[start_idx:end_idx] = label_value

        if filter_to_apply is not None:
            data = filter_to_apply.apply(self.raw_data)
        else:
            data = self.raw_data

        wsize = int(self.fs * segment_size_sec)
        # Cut data at the end to have segments of same size
        new_len = int(np.floor(len(data)/wsize)*wsize)
        data = data[:new_len]
        label_num = label_num[:new_len]

        if normalize:
            data = stats.zscore(data)

        # Reshape to segment
        seg_data = data.reshape((int(len(data)/wsize), wsize))
        seg_labels = label_num.reshape((int(len(data)/wsize), wsize))

        return seg_data, seg_labels
