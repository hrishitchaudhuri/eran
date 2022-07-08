import onnx 
import onnxruntime

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset

class SampleDataLoader(Dataset):
    def __init__(self, labels, images, transform=None, target_transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def permute_3(x):
    return np.transpose(x, [2, 0, 1])

datadict = '../data/data_batch_49x49_test_1'
datadict = pd.read_pickle(datadict)

X = datadict['data']
X = X.astype('float')
Y = datadict['labels']

test_data = SampleDataLoader(Y.astype(np.float32), X.astype(np.float32), transform=permute_3)
tests = DataLoader(test_data, batch_size=1, shuffle=True)

session = onnxruntime.InferenceSession('../nets/onnx/iisc/iisc_net.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

correctly_classified = 0
total_images = 0

means = [120.6799, 125.2353, 117.7029]
std = [54.2726, 53.9499, 55.2315]

for i, (image, label_) in enumerate(tests):
    print(image.shape)
    _, a, b, c = image.shape
    image = np.array(image)
    image = image.astype(np.float32) # / 255.0

    result = session.run([output_name], {input_name: image})
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)

    if dnnOutput == label_:
        correctly_classified += 1
        print("Correctly classified image: ", i)
        print("dnnOutput = ", dnnOutput)
    else:
        print("Incorrectly classified image: ", i)

    total_images += 1

print("Accuracy: ", correctly_classified / total_images)
