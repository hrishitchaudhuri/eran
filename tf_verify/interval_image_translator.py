import os
import sys

import numpy as np
import pandas as pd
import csv
import onnx
import onnxruntime
import time
import random

from read_net_file import read_onnx_net
from dataloader import SampleDataLoader

sys.path.insert(0, '../deepg/code/')
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')

from eran import ERAN
from config import config
from constraint_utils import get_constraints_for_dominant_label
from ai_milp import verify_network_with_milp

def generate_interval_image(image_size):
    image_lb = [random.randint(100, 120) for i in range(image_size)]
    image_ub = [random.randint(120, 140) for i in range(image_size)]

    return image_lb, image_ub

netname = '../nets/onnx/iisc/iisc_net.onnx'
epsilon = 2
domain = 'deeppoly'
dataset = 'iisc'

model, conv = read_onnx_net(netname)
eran = ERAN(model, is_onnx=True)

correctly_classified_images = 0
verified_images = 0
unsafe_images = 0
cum_time = 0

lbs, ubs = generate_interval_image(49 * 49 * 3)

perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(lbs, ubs, "deeppoly", config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
        
print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)

new_lb = nlb[-1]
new_ub = nub[-1]

possible_labels = []

if (new_ub[0] > new_lb[1] and new_ub[0] > new_lb[2]):
    possible_labels.append(0)

if (new_ub[1] > new_lb[0] and new_ub[1] > new_lb[2]):
    possible_labels.append(1)

if (new_ub[2] > new_lb[0] and new_ub[2] > new_lb[1]):
    possible_labels.append(2)

print(possible_labels)