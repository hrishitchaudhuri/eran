import os
import sys

import numpy as np
import pandas as pd
import csv
import onnx
import onnxruntime
import time

from read_net_file import read_onnx_net

sys.path.insert(0, '../deepg/code/')
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')

from eran import ERAN
from config import config
from constraint_utils import get_constraints_for_dominant_label
from ai_milp import verify_network_with_milp

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

def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        if config.subset == None:
            try:
                csvfile = open('../data/{}_test_full.csv'.format(dataset), 'r')
            except:
                csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
                print("Only the first 100 samples are available.")
        else:
            filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
            csvfile = open(filename, 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests

def normalize(image, means, stds, dataset):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

    elif (dataset == 'iisc'):
        count = 0
        tmp = np.zeros(49 * 49 * 3)
        for i in range(49 * 49):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1



def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]

    elif (dataset=='iisc'):
        count = 0
        tmp = np.zeros(49 * 49 * 3)
        for i in range(49 * 49):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(49 * 49 * 3):
            image[i] = tmp[i]

def permute_3(x):
    return np.transpose(x, [2, 0, 1])

netname = '../nets/onnx/iisc/iisc_net.onnx'
epsilon = 2
domain = 'deeppoly'
dataset = 'iisc'
num_pixels = 49 * 49

model, conv = read_onnx_net(netname)
eran = ERAN(model, is_onnx=True)

# means = [0]
# stds = [1]

correctly_classified_images = 0
verified_images = 0
unsafe_images = 0
cum_time = 0

boxes = [[[5, 124], [6, 78], [2, 234]]]

index = 1
correct = 0

"""
for box in boxes:
    specLB = [interval[0] for interval in box]
    specUB = [interval[1] for interval in box]
        
    normalize(specLB, means, stds, dataset)
    normalize(specUB, means, stds, dataset)
    
    hold, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, None)
    if hold:
        print('constraints hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))
        correct += 1
    else:
        print('constraints do NOT hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))

    index += 1

    print('constraints hold for ' + str(correct) + ' out of ' + str(sum([1 for b in boxes])) + ' boxes')


"""
constraints = None
# tests = get_tests(dataset, False)
datadict = '../data/data_batch_49x49_test_1'
datadict = pd.read_pickle(datadict)

X = datadict['data']
X = X.astype('float')
Y = datadict['labels']

means = [120.6799, 125.2353, 117.7029]
std = [54.2726, 53.9499, 55.2315]
target = []

test_data = SampleDataLoader(Y.astype(np.float32), X.astype(np.float32), transform=permute_3)
tests = DataLoader(test_data, batch_size=1, shuffle=True)

session = onnxruntime.InferenceSession('../nets/onnx/iisc/iisc_net.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

for i, (image_, label_) in enumerate(tests):
    image = np.float64(image_).transpose(0, 2, 3, 1) # / np.float64(255)

    specLB = np.copy(image)
    specUB = np.copy(image)

    image_ = np.array(image_)
    image_ = image_.astype(np.float32)

    result = session.run([output_name], {input_name: image_})
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)

    print(np.array(result).squeeze())

    # normalize(specLB, means, stds, dataset)
    # normalize(specUB, means, stds, dataset)

    is_correctly_classified = False
    start = time.time()

    label, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, "deeppoly", config.timeout_lp, config.timeout_milp, config.use_default_heuristic)

    print("concrete ", nlb[-1])
    print("concrete ", nub[-1])
    print("label", label, "of", label_, "dnn", dnnOutput)

    if label == int(label_):
        is_correctly_classified = True

    if is_correctly_classified == True:
        label = int(label_)
        perturbed_label = None
        
        correctly_classified_images += 1

        specLB = specLB - epsilon
        specUB = specUB + epsilon

        if config.quant_step:
            specLB = np.round(specLB/config.quant_step)
            specUB = np.round(specUB/config.quant_step)

        perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                            config.timeout_lp,
                                                                            config.timeout_milp,
                                                                            config.use_default_heuristic,
                                                                            label=label, prop=-1, K=0, s=0,
                                                                            timeout_final_lp=config.timeout_final_lp,
                                                                            timeout_final_milp=config.timeout_final_milp,
                                                                            use_milp=False,
                                                                            complete=False,
                                                                            terminate_on_failure=not config.complete,
                                                                            partial_milp=0,
                                                                            max_milp_neurons=0,
                                                                            approx_k=0)
        
        print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)

        if not (perturbed_label == label):
            perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, domain,
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=-1,
                                                                                      K=config.k, s=config.s,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=config.use_milp,
                                                                                      complete=config.complete,
                                                                                      terminate_on_failure=not config.complete,
                                                                                      partial_milp=config.partial_milp,
                                                                                      max_milp_neurons=config.max_milp_neurons,
                                                                                      approx_k=config.approx_k)
            print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_labels)            
    
        if (perturbed_label==label):
            print("img", i, "Verified", label)
            verified_images += 1
        else:
            if failed_labels is not None:
                failed_labels = list(set(failed_labels))
                constraints = get_constraints_for_dominant_label(label, failed_labels)
                
                verified_flag, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                if (verified_flag==True):
                    print("img", i, "Verified as Safe using MILP", label)
                    verified_images += 1
                
                else:
                    if adv_image != None:
                        cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                        if (cex_label != label):
                            denormalize(adv_image[0], means, std, dataset)
                            print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                            unsafe_images+=1
                        else:
                            print("img", i, "Failed with MILP, without a adeversarial example")
                    else:
                        print("img", i, "Failed with MILP")

        end = time.time()
        cum_time += end - start
    else:
        print("Image", i, "not considered; incorrectly classified.")
        end = time.time()

    print(f"progress: {1 + i - config.from_test}/{config.num_tests}, "
              f"correct:  {correctly_classified_images}/{1 + i - config.from_test}, "
              f"verified: {verified_images}/{correctly_classified_images}, "
              f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
              f"time: {end - start:.3f}; {0 if cum_time==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")


print('analysis precision ',verified_images,'/ ', correctly_classified_images)