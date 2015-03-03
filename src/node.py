"""A base node class representing nodes in a network

E.g. Neuron populations, input nodes, output nodes
"""
class Node(object):

  """Base constructor for all Node children class
  """
  def __init__(self, name):
    self.name = name

  """Returns the output of the node

  Note: setInput() must be called prior to this function call unless
    it is an input node
  """
  def SetInput(self, input_signal):
    raise NotImplementedError

  def GetOutput(self):
    raise NotImplementedError

class InputNode(Node):

  def __init__(self, name, input_signal):
    self.name = name
    self.SetInput(input_signal)

  def SetInput(self, input_signal):
    self.input_signal = input_signal

  def GetOutput(self):
    return self.input_signal
