from src.rc_digehealth_test.utils import DigeHealthFile
from src.rc_digehealth_test.preprocessors import ButterFilter, WaveletFilter
import torch
from torch.utils.data import WeightedRandomSampler
from src.rc_digehealth_test.classifiers import BowelSoundCNN, train_model, SegmentDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Get the files
train_file = DigeHealthFile('../test-data/AS_1.wav', decimate_data=True)
test_file = DigeHealthFile('../test-data/23M74M.wav', decimate_data=True)

# Prepare preprocessor
fs = train_file.fs
butter_filter = ButterFilter(cutoff=80, btype='highpass', fs=fs, axis=0)
wavelet_filter = WaveletFilter(fs)
# Get the preprocessed and segmented data from the files
seg_train_data, seg_train_labels = train_file.get_segmented_data(segment_size_sec=0.06, filter_to_apply=butter_filter)
seg_test_data, seg_test_labels = test_file.get_segmented_data(segment_size_sec=0.06, filter_to_apply=butter_filter)

# Keep the label as containing a sound or not (keep only if the full period is within a sound)
train_labels = (~np.any(seg_train_labels==0, axis=1)).astype(int)
Y_test = (~np.any(seg_test_labels==0, axis=1)).astype(int)

X_train, X_val, Y_train, Y_val = train_test_split(seg_train_data, train_labels, test_size=0.3)

# Calculate the weights for the training set
class_sample_counts = np.bincount(Y_train)
weights = 1. / class_sample_counts
sample_weights = weights[Y_train]
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

# Transform into tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
X_test = torch.from_numpy(seg_test_data).float()

# Setup learning loop parameters
batch_size = 128
num_epochs = 200
learning_rate = 0.1 # good for adelta

# Prepare datasets
train_dataset = SegmentDataset(X_train, Y_train)
val_dataset = SegmentDataset(X_val, Y_val)

# Prepare the model
model = BowelSoundCNN(fs=fs)

# Train the model
train_model(model, train_dataset=train_dataset, val_dataset=val_dataset, epochs=num_epochs,
            learning_rate=learning_rate, training_sampler=sampler, early_stop_n_epochs=20)

# Test the model on the test set
model.eval()
preds = model(X_test.to('cuda'))
preds = torch.argmax(preds, dim=1).cpu().detach().numpy()

pred_fixed = preds.copy()
for i in range(1, len(preds) - 1):
    if preds[i - 1] == 1 and preds[i + 1] == 1 and preds[i] == 0:
        pred_fixed[i] = 1

diff = np.diff(np.concatenate(([0], pred_fixed, [0])))
start_idxs = np.where(diff == 1)[0]
end_idxs = np.where(diff == -1)[0]
lengths = end_idxs - start_idxs

print(f'Accurary on Test Data: {np.sum(Y_test==preds)/len(Y_test)*100:.2f}')

test_file.show_data(preprocessor=butter_filter)
pred_xtime = np.arange(0.03, test_file.xtime[-1], 0.06)
plt.plot(pred_xtime[:-1], pred_fixed*400)
