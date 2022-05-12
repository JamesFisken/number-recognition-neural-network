import math
import numpy as np
import random
from PIL import Image
import time

start_time = time.time()
np.set_printoptions(suppress=False)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_image_rgb(folder, file, res):
    pix = []
    img = Image.open(folder + file)
    # img.show()
    for y in range(res):

        for x in range(res):
            a = img.getpixel((x, y))
            a = a[:-1]
            pix.append(a)

    for x, c in enumerate(pix):
        pix[x] = (c[0] + c[1] + c[2]) / 3 / 255
    return pix


def display_img(pixels):
    for i, l in enumerate(range(32)):
        print(pixels[i * 32:(i + 1) * 32])


# neural network starts here


hiddenlayer_size = 15  # size of a hidden layer
hiddenlayers = 5  # amount of hidden layers


class inputnode:
    def __init__(self, value, variability):
        self.variability = variability

        self.weight = [round(random.uniform(self.variability * -1, self.variability), 2) for i in range(
            hiddenlayer_size)]  # sets a random weight(-1.00, 1.00) for all the synapses spreading out from the input node

        self.value = value

        self.type = "inputnode"


class hiddenlayernode:
    position = 0

    def __init__(self, layer, position, nextweights, variability):
        self.variability = variability

        self.layer = layer  # is set to the hidden layer that the neuron is located in
        self.position = position  # is set to the position of a neuron in a dedicated layer
        self.nextweights = nextweights

        self.weight = [round(random.uniform(self.variability * -1, self.variability), 2) for i in range(nextweights)]
        self.bias = round(random.uniform(self.variability * -1, self.variability), 2)

        self.value = 0  # math will determine this

        self.type = "hiddenlayernode"


class outputnode:
    def __init__(self, number):
        self.number = number
        self.value = 0  # confidence that the ai has in its decision

        self.type = "outputnode"


def init(cost):
    global inputs, hiddenlayer, outputlayer, last_hiddenlayer
    inputs = []  # holds all iterations of the inputnode class and with that their weights
    hiddenlayer = []  # holds all iterations of the hiddenlayernode class
    outputlayer = []  # holds all iterations of the outputnode class

    for i in inputs_array: inputs.append(
        inputnode(i, cost))  # creates all input neurons as an instance of the inputnode class and adds them to a list

    for x in range(hiddenlayers):
        for y in range(hiddenlayer_size):
            hiddenlayer.append(hiddenlayernode(x, y, hiddenlayer_size, cost))

    for y in range(hiddenlayer_size):
        hiddenlayer.append(
            hiddenlayernode(hiddenlayers, y, 10, cost))  # prepares a hiddenlayer that only gives out 10 weights for the

    for i in range(10):  # getting all digits 1-10
        outputlayer.append(outputnode(i))


def getcost(actual_node, expected_value):
    running_total = 0
    actual_values = []
    for node in actual_node:
        actual_values.append(node.value)
    for a, i in enumerate(actual_values):
        if a == expected_value:
            running_total += (i - 1) ** 2
        else:
            running_total += (i - 0) ** 2
    return running_total


def matrix_multiply(value, weights, bias):
    final_values = []
    values = np.dot(weights, value)  # columns(1) must equal rows(2)
    values = np.add(values, bias)
    for i in values:
        final_values.append(sigmoid(i))
    return final_values


def setuparrays(layer, num_of_next_weights):
    bias = np.array([0.0 for i in range(num_of_next_weights)])
    values = np.array([0.0 for i in range(hiddenlayer_size)])

    temp_weights = np.array([a.weight for i, a in enumerate(hiddenlayer) if a.layer == layer])  # ended here
    weights = []
    for node in hiddenlayer:
        if node.layer == layer:
            values[node.position] = node.value
        if node.layer == (layer + 1):
            bias[node.position] = node.bias

    for x in range(num_of_next_weights):
        for y in range(hiddenlayer_size):
            weights.append(temp_weights[y][x])
    weights = np.reshape(weights, (num_of_next_weights, hiddenlayer_size))

    return values, weights, bias


def apply_values(layer, final_values):
    for node in hiddenlayer:
        if node.layer == layer:
            node.value = final_values[node.position]


def neuralnetwork(inputlayer, hiddenlayer, outputlayer):
    layer = 0
    bias = np.array([hiddenlayer[i].bias for i in range(hiddenlayer_size)])
    input_values = np.array(
        [inputlayer[i].value for i, a in enumerate(inputlayer)])  # creates an array with all the input layer values

    temp_weights = np.array(
        [inputlayer[i].weight for i, a in enumerate(inputlayer)])

    weights = []
    for x in range(hiddenlayer_size):
        for y in range(len(inputlayer)):
            weights.append(temp_weights[y][x])
    weights = np.reshape(weights, (hiddenlayer_size, len(inputlayer)))
    all_values = matrix_multiply(input_values, weights, bias)
    apply_values(0, all_values)
    for node in hiddenlayer:
        print(node.value)

    for i in range(hiddenlayers):
        values = setuparrays(layer, hiddenlayer_size)
        apply_values(layer + 1, matrix_multiply(values[0], values[1], values[2]))
        for node in hiddenlayer:
            print(node.value)
        layer += 1
    print("______________")

    values = setuparrays(layer, 10)
    output = matrix_multiply(values[0], values[1], values[2])
    print(output)

    for i, node in enumerate(outputlayer):
        node.value = output[i]
    for node in outputlayer:
        print(node.value)


number = 2
file = str(number) + ".png"

pixels = get_image_rgb("images/", file, 32)
inputs_array = np.array(pixels)

init(1)
for x in range(1):
    inputs = []
    number = 2
    file = str(number) + ".png"

    pixels = get_image_rgb("images/", file, 32)
    inputs_array = np.array(pixels)

    for i in inputs_array:
        inputs.append(inputnode(i, 0.2))

    for x in range(len(hiddenlayer)):
        hiddenlayer[x].value = 0  # resets hiddenlayers

    neuralnetwork(inputs, hiddenlayer, outputlayer)

    cost = getcost(outputlayer, number)

print("code took: ", time.time() - start_time, "seconds to run")