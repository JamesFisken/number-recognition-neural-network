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
  #img.show()
  for y in range(res):

    for x in range(res):
      a = img.getpixel((x, y))
      a = a[:-1]
      pix.append(a)

  for x, c in enumerate(pix):
    pix[x] = (c[0]+c[1]+c[2])/3/255
  return pix

def display_img(pixels):
  for i, l in enumerate(range(32)):
    print(pixels[i*32:(i+1)*32])



#neural network starts here
pixels = get_image_rgb("images/", "2.png", 32)
inputs_array = np.array(pixels)


hiddenlayer_size = 20 #size of a hidden layer
hiddenlayers = 5 #amount of hidden layers




class inputnode:
  def __init__(self, value):
    self.weight = [round(random.uniform(-1.0, 1.0), 2) for i in range(hiddenlayer_size)] #sets a random weight(-1.00, 1.00) for all the synapses spreading out from the input node

    self.value = value

    self.type = "inputnode"

class hiddenlayernode:
  position = 0

  def __init__(self, layer, position):

    self.layer = layer #is set to the hidden layer that the neuron is located in
    self.position = position #is set to the position of a neuron in a dedicated layer

    self.weight = [round(random.uniform(-1.0, 1.0), 2) for i in range(hiddenlayer_size)]
    self.bias = round(random.uniform(-1.0, 1.0), 2)

    self.value = 0 #math will determine this

    self.type = "hiddenlayernode"


class outputnode:
  def __init__(self, number):
    self.number = number
    self.value = 0 #confidence that the ai has in its decision

    self.type = "outputnode"

def init():
  global inputs, hiddenlayer, outputlayer
  inputs = []  # holds all iterations of the inputnode class and with that their weights
  hiddenlayer = [] #holds all iterations of the hiddenlayernode class
  outputlayer = [] #holds all iterations of the outputnode class

  for i in inputs_array: inputs.append(inputnode(i)) #creates all input neurons as an instance of the inputnode class and adds them to a list

  for x in range(hiddenlayers):
    for y in range(hiddenlayer_size):
      hiddenlayer.append(hiddenlayernode(x, y))

  for i in range(10): #getting all digits 1-10
    outputlayer.append(outputnode(i))

def matrixMultiply(layer, hiddenlayer):

  previous_values = np.array([0.0 for i in range(round((len(hiddenlayer))/hiddenlayers))])
  final_values = []
  weights = np.array([[0 for i in range(hiddenlayer_size)] for i in range(round((len(hiddenlayer))/hiddenlayers))])
  bias = np.array([0 for i in range(len(hiddenlayer))])


  for node in hiddenlayer:
    if node.layer == layer:
      weights[node.position] = node.weight #this needs to be node.weight

      previous_values[node.position] = node.value


    if node.layer == layer+1:
      bias[node.position] = node.bias #gets the bias from the hiddenlayer that we are trying to get the values for

  print("weights: ",previous_values)
  # reshapes everything so that its ready for matrix multiplication
  weights = np.reshape(weights, (20, 1024))
  previous_values = np.reshape(inputs, (1024, 1))

  values = np.matmul(weights, inputs)  # columns(1) must equal rows(2)
  values = np.add(values, bias)
  for x in values:
    final_values.append(sigmoid(x))

  for x, i in enumerate(final_values):
    print(x, i)
    hiddenlayer[x].value = i

def neuralnetwork(inputlayer, hiddenlayer, outputlayer):
  layer = 0
  weights_temp = []
  final_values = []
  bias = []
  for x in range(len(inputlayer)):
    for y in range(len(inputlayer[x].weight)):
      weights_temp.append(inputlayer[x].weight[y])
  for node in hiddenlayer:
    if node.layer == layer:
      bias.append(node.bias)

  inputs = np.array([inputlayer[i].value for i in range(len(inputlayer))])
  weights = np.array(weights_temp)
  bias = np.array(bias)

  bias = np.reshape(bias, (20, 1))  # reshapes everything so that its ready for matrix multiplication
  weights = np.reshape(weights, (20, 1024))
  inputs = np.reshape(inputs, (1024, 1))

  values = np.matmul(weights, inputs)  # columns(1) must equal rows(2)
  values = np.add(values, bias)
  for x in values:
    final_values.append(sigmoid(x))

  for x, i in enumerate(final_values):
    hiddenlayer[x].value = i

  for i in range(hiddenlayers):
    matrixMultiply(layer, hiddenlayer)




init()
neuralnetwork(inputs, hiddenlayer, outputlayer)



#---- bug testing ----
#for i in range(len(inputs)):
  #print(inputs[i].type, i, inputs[i].weight) #prints all inputs and the weights of their synapses

#for i in range(len(hiddenlayer)):
  #print(hiddenlayer[i].type, "layer: ", hiddenlayer[i].layer, "position: ", hiddenlayer[i].position, hiddenlayer[i].weight) #prints all hiddenlayer nodes and their relevant infomation

#for i in range(len(outputlayer)):
  #print(outputlayer[i].type, outputlayer[i].number, outputlayer[i].value) #prints all outputs and their values



print("code took: ", time.time() - start_time, "seconds to run")