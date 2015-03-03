import numpy as np

"""
A simulator that performs utility functions simulating multiple neuron activities
"""
class Simulator:
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
    # Gamma = np.dot(A.T, A)
    # Upsilon = np.dot(A.T, x)
    # decoders = np.dot(np.linalg.pinv(Gamma), Upsilon)

    # Use numpy built in least square solver to improve performance
    decoders, residuals2, rank, s = np.linalg.lstsq(A, x)

    return decoders

  @staticmethod
  def FilterResponses(responses, filter_kernel):
    filtered_spikes = np.array([
        np.convolve(spikes, filter_kernel, mode='same')
          for voltage, spikes in responses]).T
    return filtered_spikes

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
    filtered_spikes = Simulator.FilterResponses(responses, filter_kernel)
    return Simulator.GetDecoders(x, filtered_spikes, noise_std, transformation)

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
