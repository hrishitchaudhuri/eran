import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow.python.tools import saved_model_utils

cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')

from eran import ERAN
from config import config

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

""" def load_model(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph """

netname = "./net_testing/saved_model.pb"
assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
filename, file_extension = os.path.splitext(netname)

is_pb_file = (file_extension==".pb")

dataset = 'mnist'
non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

netfolder = os.path.dirname(netname)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
with tf.io.gfile.GFile(netname, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.graph_util.import_graph_def(graph_def, name='')
"""

graph = tf.saved_model.load(netfolder) # load_model(netname)
sess = tf.compat.v1.Session(graph=graph)

ops = sess.graph.get_operations()
last_layer_index = -1

while ops[last_layer_index].type in non_layer_operation_types:
    last_layer_index -= 1

model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')
eran = ERAN(model, sess)

means = [0]
stds = [1]

boxes = [[5, 124], [6, 78], [2, 234]]

index = 1
correct = 0

domain = 'deeppoly'

for box in boxes:
    specLB = [interval[0] for interval in box]
    specUB = [interval[1] for interval in box]
        
    normalize(specLB, means, stds, dataset)
    normalize(specUB, means, stds, dataset)
    
    hold, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
    if hold:
        print('constraints hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))
        correct += 1
    else:
        print('constraints do NOT hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))

    index += 1

    print('constraints hold for ' + str(correct) + ' out of ' + str(sum([1 for b in boxes])) + ' boxes')