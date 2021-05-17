from mnist import MNIST
import numpy as np


def load_files():
    # train_size = 60000
    # test_size = 10000

    data = MNIST('dane')

    train_data, train_res = data.load_training()
    test_data, test_res = data.load_testing()

    return data, train_data, train_res, test_data, test_res


def build_net(lay1_size, lay2_size, lay3_size):
                            # columns   # rows
    wages1 = np.random.rand(lay2_size, lay1_size)
    wages2 = np.random.rand(lay3_size, lay2_size)
    return [wages1, wages2]


def train_net(net, input_data, res):
    wages1 = net[0] #wages[c][r]
    wages2 = net[1]



    for i in range(len(input_data)):
        wages1@input_data[i] = hidden
        wages2@hidden = output[i]

    # TODO compare output with res
    # error back propagation? 
    # + b

    return [wages1, wages2]



def script():

    INPUT_LAYER_SIZE = 784
    HIDDEN_LAYER_SIZE = 20
    OUTPUT_LAYER_SIZE = 10

    set, training_images, training_labels, test_images, test_labels =  load_files()
    print(set.display(training_images[65]))

    network = build_net(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
    trained_net = train_net(network, training_images, training_labels)
    print('aa')





script()