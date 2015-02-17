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
  @staticmethod
  def GetDecoders(x, A, noise_std=None, transformation=lambda x: x):
    if noise_std:
      A = A + np.random.normal(scale=noise_std * np.max(A), size=A.shape)
    x = transformation(x)
    Gamma = np.dot(A.T, A)
    Upsilon = np.dot(A.T, x)
    #decoders = np.dot(np.linalg.pinv(Gamma), Upsilon)
    # Use numpy built in least square solver to improve performance
    decoders, residuals2, rank, s = np.linalg.lstsq(A, x)

    return decoders

  """Calcualtes the optimal temporal decoder

  Calculations:
    minimizing the square error
    E = ( X(w) - X_hat(w) )^2
    E = ( X(w) - R(w)H(w) )^2
    0 = X(w) - R(w)H(w)
    H(w) = X(w)R*(w) / |R(w)|^2

    Cutting signals into small pieces and average over =>
    convolution with W(w) in both denominator and numerator
    H(w) = X(w)R*(w) conv W(w) / |R(w)|^2 conv W(w)

  Params:
    input_signal - signal
    response_sum - sum of all neuron responses
    dt - time step
    sigma_t - Gaussian window (rate code / timing code)

  Returns:
    h, H - Optimal decoder for the input signal in time and frequency
  """
  @staticmethod
  def GetTemporalOptimalDecoder(input_signal, response_sum, dt, sigma_t=0.025):
    R = np.fft.fftshift(np.fft.fft(response_sum))
    freq = np.fft.fftshift(np.fft.fftfreq(len(R), d=dt))
    omega = freq * 2 * np.pi
    X = np.fft.fftshift(np.fft.fft(input_signal))
    W2 = np.exp(-omega ** 2 * sigma_t ** 2)
    W2 = W2 / sum(W2)
    CP = X * R.conjugate()
    WCP = np.convolve(CP, W2, 'same')
    RP = R * R.conjugate()
    WRP = np.convolve(RP, W2, 'same')
    XP = X * X.conjugate()
    H = WCP / WRP
    h = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H))).real

    return h, H

  """Get the biological plausible filter which models post synaptic current model

  h(t) = t^n * e^(-t/tau)

  Params:
    time - array of time e.g [0, 0.01, ... 0.99, 1]
    tau - neurotransmitter time constant
    n - some parameter

  Returns:
    Normalized post synaptic current filter
  """
  @staticmethod
  def GetPostSynapticFilter(time, tau, n=0):
    time = time - np.round(time[-1 * (len(time) % 2) + -1] / 2)
    h = time ** n * np.exp(-time / tau)
    h[np.where(time < 0)] = 0

    return h / np.sum(h)
