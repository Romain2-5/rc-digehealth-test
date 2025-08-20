from src.rc_digehealth_test.utils import DigeHealthFile, find_nonzero_segments, fill_label_gaps
from src.rc_digehealth_test.preprocessors import ButterFilter, WaveletFilter
import torch
from torch.utils.data import WeightedRandomSampler
from src.rc_digehealth_test.classifiers import BowelSoundCNN, train_model, SegmentDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd


# Get the files
train_file = DigeHealthFile('../test-data/AS_1.wav', decimate_data=True)
test_file = DigeHealthFile('../test-data/23M74M.wav', decimate_data=True)

# Prepare preprocessor
fs = train_file.fs
butter_filter = ButterFilter(cutoff=80, btype='highpass', fs=fs, axis=0)

# Get the preprocessed and segmented data from the large file to use as training
seg_train_data, train_labels, _ = train_file.get_segmented_data(segment_size_sec=0.06, hop=0.01, filter_to_apply=butter_filter)

# Get test dataset from the other file to use as test
seg_test_data, Y_test, test_xtime = test_file.get_segmented_data(segment_size_sec=0.06, hop=0.01, filter_to_apply=butter_filter)

# Divide training set in train and val
X_train, X_val, Y_train, Y_val = train_test_split(seg_train_data, train_labels, test_size=0.3, random_state=7,
                                                  stratify=train_labels)

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
num_epochs = 100
learning_rate = 0.1 # in the paper they use 1, I reduce because there's not a lot of data

# Prepare datasets
train_dataset = SegmentDataset(X_train, Y_train)
val_dataset = SegmentDataset(X_val, Y_val)

# Prepare the model with 3 classes output (0 = noise, 1 = burst, 2 = harmonic)
model = BowelSoundCNN(fs=fs, num_classes=3, dropout=0.3)

# Train the model
train_model(model, train_dataset=train_dataset, val_dataset=val_dataset, epochs=num_epochs,
            learning_rate=learning_rate, training_sampler=sampler, early_stop_n_epochs=20)

# Test the model on the test set and print classification report
model.eval()
preds = model(X_test.to('cuda'))
preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
print(f'Accurary on Test Data: {np.sum(Y_test==preds)/len(Y_test)*100:.2f}')
print(classification_report(Y_test, preds))

# Fix the prediction by filling gaps (use timing information)
pred_fixed = preds.copy()
pred_fixed = fill_label_gaps(pred_fixed, max_gap=7)

# Find the continuous periods to mimic the labelling style of the file
periods = find_nonzero_segments(pred_fixed)

# Consider one burst class if difference start and stop is short
for i, p in enumerate(periods):
    if p[2] == 1:
        if p[1] - p[0] < 8:
            p[2] = 0

# Save the periods in a file with same style as label file
label_map = {0: 'b', 1: "mb", 2: 'h'}
result = pd.DataFrame(periods, columns=['start', 'stop', 'label'])
result['label'] = result['label'].map(label_map)

# Show data, label and classification results
test_file.show_data(preprocessor=butter_filter)
colors = ['tab:blue', 'tab:green', 'tab:red']
for p in periods:
    plt.plot([test_xtime[p[0]], test_xtime[p[1]]], [0, 0], color=colors[p[2]], linewidth=5)

# Show confusion matrix
cm = confusion_matrix(Y_test, pred_fixed, normalize='true')
plt.figure()
sns.heatmap(cm, annot=True)

# Save model with ONNX
example_inputs = (torch.randn(1, 480),)
onnx_program = torch.onnx.export(model.to('cpu'), example_inputs, dynamo=True)
onnx_program.save("../saved_models/BowelSoundCNN.onnx")
