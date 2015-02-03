import utils

import numpy as np
import pylab


def Question1_1():
  # a
  T = 1
  dt = 0.001
  rms = 0.5
  x = np.linspace(0, 1, 1 / 0.001 + 1)
  pylab.plot(x, utils.GetGaussianWhiteNoise(T, dt, rms, 5))
  pylab.plot(x, utils.GetGaussianWhiteNoise(T, dt, rms, 10))
  pylab.plot(x, utils.GetGaussianWhiteNoise(T, dt, rms, 20))
  pylab.legend(['limit 5 Hz', 'limit 10 Hz', 'limit 20 Hz'])

  # b
  trials = 100
  coefficients = np.zeros(shape=(trials, T / dt + 1), dtype=complex)
  for i in xrange(trials):
    coefficients[i] = utils.GetCoefficientsForRealSignal(T, dt, 10)
  norm = np.mean(np.absolute(coefficients), axis=0)
  pylab.figure()
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


if __name__ == "__main__":
  Question1_1()
  Question1_2()