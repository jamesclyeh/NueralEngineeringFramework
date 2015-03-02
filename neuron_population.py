import neuron
import numpy as np
import random

"""
A simulator that performs utility functions simulating multiple neuron activities
"""
class NeuronPopulation:
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
  def __init__(self, num_neurons, neuron_model, max_firing_rate_range, \
      x_intercept_range, encoder_choices=None, encoders=None):
    self.neuron_model = neuron_model
    self.num_neurons = num_neurons
    self.max_firing_rate_range = max_firing_rate_range
    self.x_intercept_range = x_intercept_range
    self.neurons = []
    self.encoder_choices = encoder_choices
    self.encoders = encoders
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
    for i in xrange(self.num_neurons):
      x_intercept = random.uniform(*self.x_intercept_range)
      max_firing_rate = random.uniform(*self.max_firing_rate_range)
      encoder = encoders[i] if encoders else random.choice(self.encoder_choices)
      self.neurons.append((x_intercept, max_firing_rate, encoder))

  """Simulate the response of N neurons

  Params:
    values - list of values to be encoded
    n - number of neurons to be simulated

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
          max_firing_rate, x_intercept, x[-1])
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
      self, x, index=None, encoders=None, max_x=1):
    responses = []
    indices = [index] if index else range(len(self.neurons))
    for i in indices:
      x_intercept, max_firing_rate, encoder = self.neurons[i]
      alpha, J_bias = self.neuron_model.GetGainAndBias(max_firing_rate, x_intercept, max_x)
      if encoders:
        for e in encoders:
          J = alpha * x * e + J_bias
          responses.append(self.neuron_model.GetTemporalResponse(1, 0.001, J))
      else:
        J = alpha * x * encoder + J_bias
        responses.append(self.neuron_model.GetTemporalResponse(1, 0.001, J))

    return responses


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

