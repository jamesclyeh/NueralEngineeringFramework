import neuron
import numpy as np
import pylab
import random

"""
A simulator that performs utility functions simulating multiple neuron activities
"""
class Simulator:
  """Init function for the simulator

  Params:
    values - list of values to be encoded
    max_firing_rate_range - max neuron firing rate range, tuple of 2 elements
    x_intercept_range - x intercept range for different neurons
    encoder_choices - possible preferred vector direction to use, will be
      selected by random
    encoders - list encoders to use
  """
  def __init__(self, neuron_model, values, max_firing_rate_range, \
      x_intercept_range, encoder_choices=None, encoders=None):
    self.n = neuron_model
    self.values = values
    self.max_firing_rate_range = max_firing_rate_range
    self.x_intercept_range = x_intercept_range
    self.encoder_choices = encoder_choices
    self.encoders = encoders
    assert self.encoders is not None  or \
           self.encoder_choices is not None

  def SetInput(self, values):
    self.values = values

  """Simulate the response of N neurons

  Params:
    n - number of neurons to be simulated

  Returns:
    x - the values being encoded
    responses - the neuron responses for the values encoded
  """
  def GetNeuronResponses(self, n):
    num_points = self.values.shape[1]
    x = self.values
    response = []
    for i in xrange(n):
      if self.encoder_choices:
        e = [random.choice(self.encoder_choices)]
      else:
        e = self.encoders[i]
      x_intercept = random.uniform(*self.x_intercept_range)
      max_firing_rate = random.uniform(*self.max_firing_rate_range)
      gain, bias = self.n.GetGainAndBias(max_firing_rate, x_intercept)
      response.append(self.n.GetFiringRates(x, gain, bias, e))
    responses = np.vstack(tuple(response))

    return x, responses.T

  """Calculates the optimal decoder given x and neuron firing activities

  Params:
  x - information being encoded
  A - neuron activity matrix
  noise - std dev of the expected noise

  Returns:
  d - decoders
  """
  def GetDecoders(
      self, x, A, noise_std=None, transformation=lambda x: x):
    if noise_std:
      A = A + np.random.normal(scale=noise_std * np.max(A), size=A.shape)
    x = transformation(x)
    Gamma = np.dot(A.T, A)
    Upsilon = np.dot(A.T, x)
    #decoders = np.dot(np.linalg.pinv(Gamma), Upsilon)
    # Use numpy built in least square solver to improve performance
    decoders, residuals2, rank, s = np.linalg.lstsq(A, x)

    return decoders
