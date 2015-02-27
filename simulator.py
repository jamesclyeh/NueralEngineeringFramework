import neuron
import numpy as np
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
    self.neurons = []
    self.encoder_choices = encoder_choices
    self.encoders = encoders
    assert self.encoders is not None  or \
           self.encoder_choices is not None

  """Generate random neurons based on the params used to initialize
  the simulator

  Params:
    num_neurons - number of neurons to create
    encoder_choices - if specified, one of these will be randomly selected
      when generating each neuron
    encoders - if specified, encoder will be used in the order it is passed in
  Returns:
    None, access the neurons created by accessing self.neurons
  """
  def GenerateRandomNeurons(
      self, num_neurons, encoder_choices=None, encoders=None):
    for i in xrange(num_neurons):
      x_intercept = random.uniform(*self.x_intercept_range)
      max_firing_rate = random.uniform(*self.max_firing_rate_range)
      encoder = encoders[i] if encoders else random.choice(self.encoder_choices)
      self.neurons.append((x_intercept, max_firing_rate, encoder))

  def SetInput(self, values):
    self.values = values

  """Simulate the response of N neurons

  Params:
    n - number of neurons to be simulated

  Returns:
    x - the values being encoded
    responses - the neuron responses for the values encoded
  """
  def GetNeuronResponses(self, n, noise_std=None):
    num_points = self.values.shape[1]
    x = self.values
    response = []
    if not self.neurons:
      self.GenerateRandomNeurons(n)
    for i in xrange(n):
      x_intercept, max_firing_rate, e = self.neurons[i]
      gain, bias = self.n.GetGainAndBias(
          max_firing_rate, x_intercept, x[-1])
      response.append(self.n.GetFiringRates(x, gain, bias, e))
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
  #TODO: Figure out a better way to pass in encoders
  def GetTemporalNeuronResponses(
      self, x, index=None, encoders=None, max_x=1):
    responses = []
    indices = [index] if index else range(len(self.neurons))
    for i in indices:
      x_intercept, max_firing_rate, encoder = self.neurons[i]
      alpha, J_bias = self.n.GetGainAndBias(max_firing_rate, x_intercept, max_x)
      if encoders:
        for e in encoders:
          J = alpha * x * e + J_bias
          responses.append(self.n.GetTemporalResponse(1, 0.001, J))
      else:
        J = alpha * x * encoder + J_bias
        responses.append(self.n.GetTemporalResponse(1, 0.001, J))

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

  """Calculate decoders given input and temporal response

  Params:
    x - the information encoded
    fitler_kernel - filter used to perform convolution
      e.g. post synaptic current filter, optimal filter
    noise_std - the estimated noise standard deviation
    transformation - the transformation applied to input
  Returns:
    filtered_spikes - the filtered A matrix (activity matrix)
    decoders - decoders to decode filtered_spikes to x
  """
  @staticmethod
  def GetDecodersForTemporalResponses(
      x, responses, filter_kernel, noise_std=None, transformation=lambda x:x):
    filtered_spikes = np.array([
        np.convolve(spikes, filter_kernel, mode='same')
          for voltage, spikes in responses]).T
    return filtered_spikes, Simulator.GetDecoders(x, filtered_spikes)

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
    time = time - time[-1 * (len(time) % 2) + -1] / 2.
    h = time ** n * np.exp(-time / tau)
    h[np.where(time < 0)] = 0

    return h / np.sum(h), time
