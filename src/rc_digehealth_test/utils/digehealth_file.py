from scipy.io import wavfile
from scipy.signal import decimate
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from typing import Tuple, Optional
from ..preprocessors import Filter


class DigeHealthFile:
    """
    Class to load, preprocess, and segment DigeHealth bowel sound recordings.
    """

    def __init__(self, digehealth_file_path: str, decimate_data: bool = False) -> None:
        """
        Initialize a DigeHealthFile object.

        Args:
            digehealth_file_path (str): Path to the .wav file.
            decimate_data (bool, optional): If True, decimate data to 8 kHz sampling rate.
                Defaults to False.
        """

        self.digehealth_file_path = digehealth_file_path
        original_fs, self.raw_data = wavfile.read(digehealth_file_path)

        if decimate_data:
            self.raw_data = decimate(self.raw_data, int(original_fs / 8000))
            self.fs = 8000
        else:
            self.fs = original_fs

        self.xtime = np.arange(0, len(self.raw_data) / self.fs, 1 / self.fs)

        if os.path.exists(digehealth_file_path.replace('.wav', '.txt')):
            self.labels = pd.read_csv(digehealth_file_path.replace('.wav', '.txt'), header=0, sep='\t',
                                    names=['start', 'end', 'label'])
            self.labels.loc[self.labels.label == 'sb', 'label'] = 'b' # it's not consistent across files.
        else:
            self.labels = None

    def get_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the raw signal data and corresponding time axis.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - xtime: Time vector in seconds.
                - raw_data: Signal samples.
        """

        return self.xtime, self.raw_data

    def get_labels(self):
        """
        Get the annotation labels.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ["start", "end", "label"].
            Returns None if no labels are available.
        """

        return self.labels

    def show_data(self, preprocessor: Optional[Filter] = None) -> None:
        """
        Plot the raw or preprocessed signal with labeled regions highlighted.

        Args:
            preprocessor (object, optional): Object with an `apply(data)` method (e.g., a filter).
                If provided, applies preprocessing to the raw signal before plotting.
        """

        xtime, data = self.get_raw_data()

        if preprocessor is not None:
            data = preprocessor.apply(data)

        fig, ax = plt.subplots()
        ax.plot(xtime, data, color='tab:grey')
        got_plot = [False, False, False]
        for i in range(len(self.labels)):
            match self.labels.iloc[i]['label']:
                case 'h':
                    color = 'tab:red'
                    igp = 0
                case 'b':
                    color = 'tab:blue'
                    igp = 1
                case 'mb':
                    color = 'tab:green'
                    igp = 2
                case _:
                    color = None
                    igp = None
            if color:
                ax.axvspan(self.labels.iloc[i]['start'], self.labels.iloc[i]['end'], color=color, alpha=0.2,
                           label=str(self.labels.iloc[i]['label']) if not got_plot[igp] else None)
                got_plot[igp] = True

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

    def get_segmented_data(
            self,
            segment_size_sec: float = 0.06,
            hop: Optional[float] = None,
            filter_to_apply=None,
            normalize: bool = True,
            group_bursts: bool = True,
            return_one_label_per_segment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment the audio data into overlapping windows (like spectrogram frames).

        Args:
            segment_size_sec (float, optional): Length of each segment in seconds. Defaults to 0.06.
            hop (float, optional): Hop size in seconds. If None, defaults to segment_size_sec (no overlap).
            filter_to_apply (Filter, optional): Filter object with .apply() method to pre-process data.
            normalize (bool, optional): If True, z-score normalization is applied to the signal. Defaults to True.
            group_bursts (bool, optional): If True, group "b" and "mb" into the same label. Defaults to True.
            return_one_label_per_segment (bool, optional): If True, return one label per segment. Otherwise, return
             label as same shape as segments Defaults to True.
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - seg_data: Array of shape (n_segments, segment_size_samples)
                - seg_labels: Array of shape (n_segments, segment_size_samples)

        """

        # Map labels to numeric
        if group_bursts:
            label_map = {"b": 1, "mb": 1, "h": 2}
        else:
            label_map = {"b": 1, "mb": 3, "h": 2}
        self.labels["label_num"] = (
            self.labels["label"].map(label_map).fillna(0).astype(int)
        )

        xtime_lookup = np.searchsorted(
            self.xtime, self.labels[["start", "end"]].values, side="left"
        )
        label_num = np.zeros_like(self.raw_data)

        for idx, (start_idx, end_idx) in enumerate(xtime_lookup):
            label_value = self.labels["label_num"].iloc[idx]
            label_num[start_idx:end_idx] = label_value

        # Filtering and normalization
        if filter_to_apply is not None:
            data = filter_to_apply.apply(self.raw_data)
        else:
            data = self.raw_data

        if normalize:
            data = stats.zscore(data)

        # Windowing
        wsize = int(self.fs * segment_size_sec)
        hop_size = int(self.fs * hop) if hop is not None else wsize

        n_segments = 1 + (len(data) - wsize) // hop_size
        seg_data = np.zeros((n_segments, wsize))
        seg_labels = np.zeros((n_segments, wsize), dtype=int)
        seg_xtime = np.zeros((n_segments, wsize))

        for i in range(n_segments):
            start = i * hop_size
            end = start + wsize
            seg_data[i] = data[start:end]
            seg_labels[i] = label_num[start:end]
            seg_xtime[i] = self.xtime[start:end]

        if return_one_label_per_segment:
            seg_labels = np.array([int(np.unique(l)[0]) for l in seg_labels])
            seg_xtime = np.mean(seg_xtime, axis=1)

        return seg_data, seg_labels, seg_xtime
