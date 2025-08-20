from src.rc_digehealth_test.utils import DigeHealthFile
from src.rc_digehealth_test.classifiers import BowelSoundCNN
from src.rc_digehealth_test.preprocessors import ButterFilter
import numpy as np
import torch
import matplotlib.pyplot as plt


file_1 = DigeHealthFile('../test-data/AS_1.wav', decimate_data=True)
# test_file = DigeHealthFile('../test-data/23M74M.wav', decimate_data=True)

butter_filter = ButterFilter(cutoff=80, btype='highpass', fs=file_1.fs, axis=0)
seg_data, lab, xtime = file_1.get_segmented_data(segment_size_sec=0.06, hop=0.1, filter_to_apply=butter_filter,
                                                    group_bursts=False, return_one_label_per_segment=True)

model = BowelSoundCNN()
spec = model.mel_spec(torch.from_numpy(seg_data).float()).numpy()
spec = np.mean(spec, axis=2)
# spec = 10 * np.log10(spec)

plt.plot(np.mean(spec[lab==0], 0), label='No noise')
plt.plot(np.mean(spec[lab==1], 0), label='One burst')
plt.plot(np.mean(spec[lab==3], 0), label='Multi burst')
plt.plot(np.mean(spec[lab==2], 0), label='Harmonic')
plt.legend()
plt.ylabel('Mel-spectrogram power (Normalized)')
plt.xlabel('Time (s)')
plt.show()
plt.xlabel('Frequency (Mels)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

plt.show()
