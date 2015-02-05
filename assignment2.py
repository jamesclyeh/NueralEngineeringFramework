import neuron
import utils

import numpy as np
import pylab


def Question1_1():
  # a
  T = 1
  dt = 0.001
  rms = 0.5
  x = np.linspace(0, 1, 1 / 0.001 + 1)
  pylab.title('Gaussian white noise with maximum frequency 5 Hz')
  pylab.plot(x, utils.GetGaussianWhiteNoise(T, dt, rms, 5))
  pylab.figure()
  pylab.title('Gaussian white noise with maximum frequency 10 Hz')
  pylab.plot(x, utils.GetGaussianWhiteNoise(T, dt, rms, 10))
  pylab.figure()
  pylab.title('Gaussian white noise with maximum frequency 20 Hz')
  pylab.plot(x, utils.GetGaussianWhiteNoise(T, dt, rms, 20))

  # b
  trials = 100
  coefficients = np.zeros(shape=(trials, T / dt + 1), dtype=complex)
  for i in xrange(trials):
    coefficients[i] = utils.GetCoefficientsForRealSignal(T, dt, 10)
  norm = np.mean(np.absolute(coefficients), axis=0)
  pylab.figure()
  pylab.title('Average norm of Fourier coefficients')
  pylab.scatter(np.fft.fftfreq(len(norm), d=dt) * 2 * np.pi, norm)
  pylab.xlim(-100, 100)
  pylab.xlabel('$\omega$ (rad/s)')
  pylab.ylabel('|X($\omega$)|')
  pylab.show()

def Question1_2():
  # a
  T = 1
  dt = 0.001
  rms = 0.5
  x = np.linspace(0, 1, 1 / 0.001 + 1)
  pylab.plot(x, utils.GetGaussianPowerSpectrumNoise(T, dt, rms, 5))
  pylab.plot(x, utils.GetGaussianPowerSpectrumNoise(T, dt, rms, 10))
  pylab.plot(x, utils.GetGaussianPowerSpectrumNoise(T, dt, rms, 20))
  pylab.legend(['limit 5 Hz', 'limit 10 Hz', 'limit 20 Hz'])

  # b
  trials = 100
  coefficients = np.zeros(shape=(trials, T / dt + 1), dtype=complex)
  for i in xrange(trials):
    coefficients[i] = utils.GetCoefficientsForRealSignalUsingGaussianPowerSpecturm(T, dt, 10)
  norm = np.mean(np.absolute(coefficients), axis=0)
  pylab.figure()
  pylab.scatter(np.fft.fftfreq(len(norm), d=dt) * 2 * np.pi, norm)
  pylab.xlim(-100, 100)
  pylab.xlabel('$\omega$ (rad/s)')
  pylab.ylabel('|X($\omega$)|')
  pylab.show()

def Question2():
  # a
  """
  40 = 1 / (tau_ref - tau_RC * ln(1 - 1 / J_bias))
  tau_ref - tau_RC * ln(1 - 1 / J_bias) = 1 / 40
  (tau_ref - 1 / 40) / tau_RC = ln(1 - 1 / J_bias)
  e^((tau_ref - 1 / 40) / tau_RC) = 1 - 1 / J_bias
  1 - e^((tau_ref - 1 / 40) / tau_RC) = 1 / J_bias
  J_bias = 1 / (1 - e^((tau_ref - 1 / 40) / tau_RC))
  alpha + J_bias = 1 / (1 - e^((tau_ref - 1 / 150) / tau_RC))
  """
  tau_ref = 0.002
  tau_RC = 0.02
  J_bias = 1 / (1 - np.exp((tau_ref - 1 / 40.0) / tau_RC))
  alpha = 1 / (1 - np.exp((tau_ref - 1 / 150.0) / tau_RC)) - J_bias
  n = neuron.LIFNeuron(tau_ref, tau_RC)
  x = np.linspace(-1, 1, 100)

  pylab.plot(x, n.GetFiringRates(x, alpha=alpha, J_bias=J_bias, e=1))
  pylab.xlabel('J (current)')
  pylab.ylabel('$a$ (Hz)')

  x = 1
  J = alpha * x + J_bias
  T = 1
  dt = 0.001
  J = np.ones(T / dt) * J
  voltage, spikes = n.GetTemporalResponse(T, dt, J)
  t = np.linspace(0, T, T / dt)
  pylab.figure()
  pylab.vlines(t * spikes, 0.5, 1.5, color='r', linewidth=1)
  pylab.plot(t, voltage)

  rms = 0.5
  max_frequency = 30
  x = np.linspace(0, T + dt, T / dt + 1)
  pylab.figure()
  pylab.title('Spikes generated with Gaussian white noise input')
  input_signal = utils.GetGaussianWhiteNoise(T, dt, rms, max_frequency)
  pylab.plot(x, input_signal)
  J = alpha * input_signal + J_bias
  voltage, spikes = n.GetTemporalResponse(T, dt, J)
  pylab.vlines(x * spikes, -0.5, 0.5, color='r', linewidth=1)

  pylab.figure()
  pylab.plot(x, voltage)
  pylab.vlines(x * spikes, 0.5, 1.5, color='r', linewidth=1)
  pylab.show()

if __name__ == "__main__":
  #Question1_1()
  #Question1_2()
  Question2()