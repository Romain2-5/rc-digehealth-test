from src.rc_digehealth_test.utils import DigeHealthFile
from src.rc_digehealth_test.preprocessors import ButterFilter
import torch
from torch.utils.data import WeightedRandomSampler
from src.rc_digehealth_test.classifiers import BowelSoundCNN, train_model, SegmentDataset
import matplotlib.pyplot as plt
import numpy as np


# Get the files
train_file = DigeHealthFile('../test-data/AS_1.wav', decimate_data=True)
test_file = DigeHealthFile('../test-data/23M74M.wav', decimate_data=True)

# Prepare preprocessor
fs = train_file.fs
butter_filter = ButterFilter(cutoff=80, btype='highpass', fs=fs, axis=0)

# Get the preprocessed and segmented data from the files
seg_train_data, seg_train_labels = train_file.get_segmented_data(segment_size_sec=0.06, filter_to_apply=butter_filter)
seg_test_data, seg_test_labels = test_file.get_segmented_data(segment_size_sec=0.06, filter_to_apply=butter_filter)

# Keep the label as containing a sound or not (keep only if the full period is within a sound)
train_labels = (~np.any(seg_train_labels==0, axis=1)).astype(int)
test_labels = (~np.any(seg_test_labels==0, axis=1)).astype(int)

# Calculate the weights for the training set
class_sample_counts = np.bincount(train_labels)
weights = 1. / class_sample_counts
sample_weights = weights[train_labels]

# Transform into tensors
X_train = torch.from_numpy(seg_train_data).float()
X_test = torch.from_numpy(seg_test_data).float()
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)
# Setup learning loop
batch_size = 128
num_epochs = 100
learning_rate = 1 # good for adelta

# Prepare datasets
train_dataset = SegmentDataset(X_train, train_labels)
test_dataset = SegmentDataset(X_test, test_labels)

# Prepare the model
model = BowelSoundCNN(fs=fs)

# Train the model
train_model(model, train_dataset=train_dataset, val_dataset=test_dataset, epochs=num_epochs,
            learning_rate=learning_rate, training_sampler=sampler)

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

test_file.show_data(preprocessor=butter_filter)
pred_xtime = np.arange(0.03, test_file.xtime[-1], 0.06)
plt.plot(pred_xtime[:-1], pred_fixed*400)
