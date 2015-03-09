from node import Node
from simulator import Simulator
from utils import GetTimeArray, GetGaussianWhiteNoise

import numpy as np
import pylab
import random
from sklearn.utils.extmath import cartesian

"""
A simulator that performs utility functions simulating multiple neuron activities
"""
class NeuronPopulation(Node):
  """Init function for the neuron population

  Params:
    num_neurons - number of neurons to create
    neuron_model - neuron model for this population
    max_firing_rate_range - max neuron firing rate range, tuple of 2 elements
    x_intercept_range - x intercept range for different neurons
    encoder_choices - possible preferred vector direction to use, will be
      selected by random
    encoders - list encoders to use
  """
  def __init__(self, name, num_neurons, neuron_model, max_firing_rate_range, \
      x_intercept_range, encoder_choices=None, encoders=None, dimensions=1):
    self.name = name
    self.neuron_model = neuron_model
    self.num_neurons = num_neurons
    self.max_firing_rate_range = max_firing_rate_range
    self.x_intercept_range = x_intercept_range
    self.neurons = []
    self.encoder_choices = encoder_choices
    self.encoders = encoders
    self.dimensions = dimensions
    self.input_signal = None
    self.transformation = None
    self.decoders = None
    assert self.encoders is not None  or \
           self.encoder_choices is not None
    self.GenerateRandomNeurons(encoder_choices, encoders)

  """Generate random neurons based on the params used to initialize
  the neuron population

  Params:
    encoder_choices - if specified, one of these will be randomly selected
      when generating each neuron
    encoders - if specified, encoder will be used in the order it is passed in
  Returns:
    None, access the neurons created by accessing self.neurons
  """
  def GenerateRandomNeurons(
      self, encoder_choices=None, encoders=None):
    self.neurons = []
    for i in xrange(self.num_neurons):
      x_intercept = random.uniform(*self.x_intercept_range)
      max_firing_rate = random.uniform(*self.max_firing_rate_range)
      encoder = encoders[i] \
          if encoders is not None else random.choice(self.encoder_choices)
      self.neurons.append((x_intercept, max_firing_rate, encoder))

  def SetNumberOfNeurons(self, num_neurons):
    self.num_neurons = num_neurons
    self.GenerateRandomNeurons(self.encoder_choices, self.encoders)

  def GenerateDecoders(self, transformation, num_points=100, noise=0.1):
    if self.transformation != transformation:
      full_x_range = np.array([np.linspace(
          self.x_intercept_range[0], self.x_intercept_range[1], num_points)]).T
      full_range_input = cartesian(
          [full_x_range.T for i in xrange(self.dimensions)])
      x, responses = self.GetNeuronResponses(full_range_input)
      responses_noisy = NeuronPopulation.AddNoiseToResponse(responses, noise)
      self.decoders = \
          Simulator.GetDecoders(x, responses_noisy, noise, transformation=transformation)
      self.transformation = transformation

  """Simulate the response of N neurons

  Params:
    values - list of values to be encoded

  Returns:
    x - the values being encoded
    responses - the neuron responses for the values encoded
  """
  def GetNeuronResponses(self, values, noise_std=None):
    num_points = values.shape[1]
    x = values
    response = []
    if not self.neurons:
      self.GenerateRandomNeurons(n)
    for i in xrange(self.num_neurons):
      x_intercept, max_firing_rate, e = self.neurons[i]
      gain, bias = self.neuron_model.GetGainAndBias(
          max_firing_rate, x_intercept, self.x_intercept_range[-1])
      response.append(self.neuron_model.GetFiringRates(x, gain, bias, e))
    responses = np.vstack(tuple(response)).T
    if noise_std:
      responses = self.AddNoiseToResponse(responses, noise_std)

    return x, responses

  """
  Get an array of temporal responses from neurons

  Params:
    x - input signal
    index - default to use all neurons in the simulator, if this param is
      specified, use the specfic neuron only
    encoders - defaults to using the encoder associated with neuron, if
      specified, use the passed in encoders instead
  Returns:
    An array containing temporal neural responses (voltage, spikes)
  """
  #FIXME: Encoders is currently used as a hack to get pair neuron responses
  def GetTemporalNeuronResponses(
      self, x, index=None, encoders=None):
    responses = []
    indices = [index] if index else range(len(self.neurons))
    for i in indices:
      x_intercept, max_firing_rate, encoder = self.neurons[i]
      alpha, J_bias = self.neuron_model.GetGainAndBias(
          max_firing_rate, x_intercept, self.x_intercept_range[-1])
      if encoders:
        for e in encoders:
          J = alpha * x * e + J_bias
          responses.append(self.neuron_model.GetTemporalResponse(1, 0.001, J))
      else:
        J = (alpha * np.dot(np.array(encoder), np.array(x).T) + J_bias).T
        responses.append(self.neuron_model.GetTemporalResponse(1, 0.001, J))

    return responses

  def SetInput(self, input_signal):
    if self.input_signal is not None:
      self.input_signal += input_signal
    else:
      self.input_signal = input_signal

  def GetOutput(self):
    if self.input_signal is None:
      raise ValueError("Node not connected to input.")
    t = GetTimeArray(1, 0.001)
    filter_kernel, kernel_time = Simulator.GetPostSynapticFilter(t, 0.005)
    responses = self.GetTemporalNeuronResponses(self.input_signal)
    filtered_spikes = Simulator.FilterResponses(responses, filter_kernel)
    #self.decoders = Simulator.GetDecodersForTemporalResponses(
    #      self.input_signal, responses, filter_kernel, transformation=self.transformation)
    self.input_signal = None

    return np.dot(filtered_spikes, self.decoders)

  """Add random normal noise to the activity matrix

  Params:
    responses - the A matrix (activity matrix
    noise_std - the standard deviation used to generate noise

  Returns:
    noisy A matrix
  """
  @staticmethod
  def AddNoiseToResponse(response, noise_std):
    return response + np.random.normal(
        scale=noise_std * np.max(response), size=response.shape)

