import math
import numpy as np
import random
from PIL import Image
import time

start_time = time.time()
np.set_printoptions(suppress=True)


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


hiddenlayer_size = 11  # size of a hidden layer
hiddenlayers = 3  # amount of hidden layers


class inputnode:
    def __init__(self, value, variability):
        self.variability = variability


        self.weight = [round(random.uniform(self.variability*-1, self.variability), 2) for i in range(
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

        self.weight = [round(random.uniform(self.variability*-1, self.variability), 2) for i in range(nextweights)]
        self.bias = round(random.uniform(self.variability*-1, self.variability), 2)

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
        hiddenlayer.append(hiddenlayernode(hiddenlayers, y, 10, cost)) #prepares a hiddenlayer that only gives out 10 weights for the


    for i in range(10):  # getting all digits 1-10
        outputlayer.append(outputnode(i))


def lastlayer(hiddenlayer_values, hiddenlayer_weights):
    weights = []
    final_values = []
    hiddenlayer_weights = np.array(hiddenlayer_weights)
    hiddenlayer_weights = np.reshape(hiddenlayer_weights, (hiddenlayer_size, 10))

    for x in range(10):
        for y in range(len(hiddenlayer_weights)):
            weights.append(hiddenlayer_weights[y][x])
    hiddenlayer_weights = np.reshape(weights, (10, hiddenlayer_size))

    values = np.reshape(hiddenlayer_values, (hiddenlayer_size, 1))
    values = np.dot(hiddenlayer_weights, hiddenlayer_values)
    for i in values:
        final_values.append(sigmoid(i))

    for a, node in enumerate(outputlayer):
        if node.number == a:
            outputlayer[a].value = final_values[a]
    for x in outputlayer:
        print("final_values", x.value)



def matrixMultiply(layer, hiddenlayer, lastlayer):
    previous_values = np.array([0.0 for i in range(round((len(hiddenlayer)-hiddenlayer_size) / hiddenlayers))])

    final_values = []
    ordered_weights = []

    if lastlayer == False:
        weights = np.array(
            [[0.0 for i in range(hiddenlayer_size)] for i in range(round((len(hiddenlayer)-hiddenlayer_size) / hiddenlayers))])
    else:
        weights = np.array(
            [[0.0 for i in range(10)] for i in range(20)])

    bias = np.array([0.0 for i in range(round((len(hiddenlayer)-hiddenlayer_size) / hiddenlayers))])

    for node in hiddenlayer:
        if node.layer == layer:
            try:
                weights[node.position] = node.weight
                previous_values[node.position] = node.value
            except:
                print("")

        if node.layer == layer + 1:
            bias[node.position] = node.bias

    for y in range(len(weights)):
        for x in range(len(weights)):
                ordered_weights.append(weights[x][y])

    weights = np.array(ordered_weights)
    weights = np.reshape(weights, (hiddenlayer_size, hiddenlayer_size))

    values = np.dot(weights, previous_values)  # columns(1) must equal rows(2)
    values = np.add(values, bias)
    for x in values:
        final_values.append(sigmoid(x))

    i = 0
    for node in hiddenlayer:
        if node.layer == layer+1:

            node.value = final_values[i]
            i += 1
    # reshapes everything so that its ready for matrix multiplication
def getcost(actual_node, expected_value):
    running_total = 0
    actual_values = []
    for node in actual_node:
        actual_values.append(node.value)
    for a, i in enumerate(actual_values):
        if a == expected_value:
            running_total += (i - 1)**2
        else:
            running_total += (i - 0)**2
    return running_total




def neuralnetwork(inputlayer, hiddenlayer, outputlayer):
    layer = 0
    weights_temp = []
    final_values = []

    bias = []
    last_hiddenlayer = []
    print(len(hiddenlayer))

    for x in range(len(inputlayer)):
        for y in range(len(inputlayer[x].weight)):
            weights_temp.append(inputlayer[x].weight[y])
    for node in hiddenlayer:
        if node.layer == layer:
            bias.append(node.bias)


    inputs = np.array([inputlayer[i].value for i in range(len(inputlayer))])
    weights = np.array(weights_temp)
    bias = np.array(bias)

    bias = np.reshape(bias, (hiddenlayer_size, 1))  # reshapes everything so that its ready for matrix multiplication
    weights = np.reshape(weights, (hiddenlayer_size, 1024))
    inputs = np.reshape(inputs, (1024, 1))

    values = np.matmul(weights, inputs)  # columns(1) must equal rows(2)
    values = np.add(values, bias)
    for x in values:
        final_values.append(sigmoid(x))


    for x, i in enumerate(final_values):
        hiddenlayer[x].value = i

    for x in hiddenlayer:
        print(x.value)

    for i in range(hiddenlayers): #might need to be changed

        matrixMultiply(layer, hiddenlayer, False)
        layer += 1
        print("")
        for x in hiddenlayer:
            print(x.value)

    for x in hiddenlayer:
        if len(x.weight) == 10:
            last_hiddenlayer.append(x) #last hidden layer

    weights = []
    values = []
    for item in last_hiddenlayer:
        weights.append(item.weight)
        values.append(item.value)
    lastlayer(values, weights)


number = 2
file = str(number) + ".png"

pixels = get_image_rgb("images/", file, 32)
inputs_array = np.array(pixels)

init(1)
for x in range(2):

    inputs = []
    number = x
    file = str(number) + ".png"

    pixels = get_image_rgb("images/", file, 32)
    inputs_array = np.array(pixels)

    for i in inputs_array:
        inputs.append(inputnode(i, 0)) #something wrong here



    for x in range(len(hiddenlayer)):
        hiddenlayer[x].value = 0
    neuralnetwork(inputs, hiddenlayer, outputlayer)

    cost = getcost(outputlayer, number)
    print(cost)


print("code took: ", time.time() - start_time, "seconds to run")