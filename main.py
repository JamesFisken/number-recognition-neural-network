import math
import numpy as np
import random
from PIL import Image
import time
import MnistDataSet

start_time = time.time()
np.set_printoptions(suppress=True)


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        if x < -500:
            return 0
        elif x > 500:
            return 1
        else:
            print(x)
            exit()


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

evolution_sample_size = 10
generations = 100
hiddenlayer_size = 30  # size of a hidden layer
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


def init(cost, inputs_array):
    global inputs, hiddenlayer, outputlayer, last_hiddenlayer
    inputs = []  # holds all iterations of the inputnode class and with that their weights
    hiddenlayer = []  # holds all iterations of the hiddenlayernode class
    outputlayer = []  # holds all iterations of the outputnode class

    for i in inputs_array: inputs.append(
        inputnode(i, cost))  # creates all input neurons as an instance of the inputnode class and adds them to a list

    for x in range(hiddenlayers):
        for y in range(hiddenlayer_size):
            hiddenlayer.append(hiddenlayernode(x, y, hiddenlayer_size, cost))

    for i in range(10):  # getting all digits 1-10
        outputlayer.append(outputnode(i))


def getcost(actual_node, expected_value):
    running_total = 0
    actual_values = []

    for node in actual_node:
        actual_values.append(node.value)
    for a, i in enumerate(actual_values):
        if a == expected_value:
            running_total += ((i - 1) ** 2)
        else:
            running_total += ((i - 0) ** 2)
    return running_total


def matrix_multiply(value, weights, bias):
    final_values = []
    values = np.matmul(weights, value)  # columns(1) must equal rows(2)
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


def adjust_modifiers(inputlayer, hiddenlayer, variability):
    for node in inputlayer:
        for i, weight in enumerate(node.weight):
            node.weight[i] += random.uniform(variability * -1, variability)

    for node in hiddenlayer:
        for i, weight in enumerate(node.weight):
            node.weight[i] += random.uniform(variability * -1, variability)
        node.bias += random.uniform(variability * -1, variability)


def apply_values(layer, final_values):
    for node in hiddenlayer:
        if node.layer == layer:
            node.value = final_values[node.position]


def print_neural_network(hiddenlayer):
    for node in hiddenlayer:
        print("layer:", node.layer, "position:", node.position, "value:", node.value, "weights:", node.weight, "bias:",
              node.bias)


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

    # print_neural_network(hiddenlayer)

    for i in range(hiddenlayers):
        values = setuparrays(layer, hiddenlayer_size)
        apply_values(layer + 1, matrix_multiply(values[0], values[1], values[2]))
        layer += 1

    values = setuparrays(layer - 1, 10)
    output = matrix_multiply(values[0], values[1], values[2])

    for i, node in enumerate(outputlayer):
        node.value = output[i]

    return getcost(outputlayer, label), outputlayer


best_results = 10 #set to a impossibly high cost
pixels, label = MnistDataSet.get_image(0) #gets pixels of a singular image and its label
init(0, pixels) #creates an empty neural network
adjust_modifiers(inputs, hiddenlayer, 1) #gives NN weights and bias variability

#saves the structure of the NN just made
save_weights = [node.weight for node in hiddenlayer]
save_input_weights = [node.weight for node in inputs]
save_bias = [node.bias for node in hiddenlayer]


F_results = 0 #so that nothing changes at the start
for x in range(generations):
    numbers = [random.randint(0, 10000) for y in range(evolution_sample_size)]  #gets a set of random numbers
    results = []

    current_save_weights = save_weights.copy()
    current_save_input_weights = save_input_weights.copy()
    current_save_bias = save_bias.copy()

    for x, weights in enumerate(current_save_input_weights):
        for y, weight in enumerate(weights):
            current_save_input_weights[x][y] += random.uniform(best_results * -5, best_results*5)

    for x, weights in enumerate(current_save_weights):
        for y, weight in enumerate(weights):
            current_save_weights[x][y] += random.uniform(best_results * -5, best_results*5)

    for i in range(len(current_save_bias)):
        current_save_bias[i] += random.uniform(best_results * -5, best_results*5)

    score = 0
    for num in numbers: #for every number in the set
        pixels, label = MnistDataSet.get_image(num)  # set pixels and labels to their corisponding number from MNIST
        init(0, pixels)  # reset all list containing nodes

        for i, node in enumerate(hiddenlayer): #retreives saved weights and bias
            node.weight = current_save_weights[i]
            node.bias = current_save_bias[i]
        for i, node in enumerate(inputs):
            node.weight = current_save_input_weights[i]
        #adjust_modifiers(inputs, hiddenlayer, 1) # this needs to be locked for each generation

        cost, output = neuralnetwork(inputs, hiddenlayer, outputlayer)
        output_num = [output[i].value for i in range(len(output))]

        if output_num.index(max(output_num)) == label:
            score += 1
        else:
            score -= 1
        results.append(cost)
    print(score)
    F_results = (sum(results)/len(results))
    if F_results < best_results:
        print("new best:", F_results)
        best_results = F_results
        save_weights = [node.weight for node in hiddenlayer]
        save_input_weights = [node.weight for node in inputs]
        save_bias = [node.bias for node in hiddenlayer]
    print("fail:", F_results)


print("code took: ", time.time() - start_time, "seconds to run")
