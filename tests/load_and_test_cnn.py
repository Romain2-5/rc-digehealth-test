import onnx
import onnxruntime
from src.rc_digehealth_test.utils import DigeHealthFile, fill_label_gaps
from src.rc_digehealth_test.preprocessors import ButterFilter
import numpy as np
from sklearn.metrics import classification_report


# Get the test data
test_file = DigeHealthFile('../test-data/23M74M.wav', decimate_data=True)
butter_filter = ButterFilter(cutoff=80, btype='highpass', fs=test_file.fs, axis=0)
x_test, seg_test_labels = test_file.get_segmented_data(segment_size_sec=0.06, filter_to_apply=butter_filter)

y_test = np.array([int(np.unique(l)[0]) for l in seg_test_labels])

# Ensure test data is float32
x_test = x_test.astype(np.float32)

# Import the saved model
onnx_model = onnx.load("../saved_models/BowelSoundCNN.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("../saved_models/BowelSoundCNN.onnx",
                                           providers=["CPUExecutionProvider"])

# Run inference on the whole dataset at once
onnxruntime_outputs = []
for i in x_test:
    onnxruntime_input = {'x': i[np.newaxis, :]}
    out = ort_session.run(None, onnxruntime_input)[0]
    onnxruntime_outputs.append(out)

# Get predicted class labels
pred_labels = np.array([int(np.argmax(out)) for out in onnxruntime_outputs])
pred_fixed = fill_label_gaps(pred_labels)

print(classification_report(y_test, pred_fixed))
