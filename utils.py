import numpy as np

"""Generates a Gaussian white noise signal

Params:
  T - duration of signal
  dt - time step of signal
  rms - RMS of signal
  limit - the maximum frequency the signal will have
  seed - seed to use when generating random numbers

Returns:
  Gaussian white noise signal
"""
def GetGaussianWhiteNoise(T, dt, rms, limit=None, seed=None):
  signal = np.fft.ifft(GetCoefficientsForRealSignal(T, dt, limit, seed)).real
  signal = signal / GetRMS(signal) * rms

  return signal

"""Generates a Gaussian power spectrum noise signal

Params:
  T - duration of signal
  dt - time step of signal
  rms - RMS of signal
  bandwidth - bandwidth of the signal

Returns:
  Gaussian power spectrum noise signal
"""
def GetGaussianPowerSpectrumNoise(T, dt, rms, bandwidth, seed=None):
  signal = np.fft.ifft(GetCoefficientsForRealSignalUsingGaussianPowerSpecturm(T, dt, bandwidth, seed)).real
  signal = signal / GetRMS(signal) * rms

  return signal

"""Generates a Fourier coefficient series (using gaussian spectrum) that will result in
a real signal when the inverse transform is performed

coeff = N(0, np.exp(-1 * (w ** 2) / (2 * ((bandwidth * 2 * np.pi) ** 2))))

The generation of the series uses the symmetric property of Fourier seires
and generates a Hermitian symmetric series.

X(w) = X(-w)*

Params:
  T - duration of signal
  dt - time step of signal
  limit - the maximum frequency of the Fourier coefficients
  seed - seed to use when generating random numbers

Returns:
  Fourier coefficient series
"""
def GetCoefficientsForRealSignalUsingGaussianPowerSpecturm(T, dt, bandwidth, seed=None):
  np.random.seed(seed)
  num_coefficients = int(T / dt / 2)
  frequencies = 2.0 * np.pi / T * (np.asarray(range(num_coefficients)) + 1)
  first_half = np.zeros(shape=(1, num_coefficients), dtype=complex)
  for index, w in enumerate(frequencies):
    sigma = np.exp(-1 * (w ** 2) / (2 * ((bandwidth * 2 * np.pi) ** 2)))
    if sigma > 0.00000000001:
      coefficient = np.random.normal(0, sigma)
      coefficient += 1j * np.random.normal(0, sigma)
    else:
      coefficient = 0
    first_half[0][index] = coefficient
  symmetric_half = np.conj(first_half[0][::-1])
  coefficients = np.zeros(shape=(1, 2 * num_coefficients + 1), dtype=complex)
  # set dc to 0
  coefficients[0][0] = 0
  coefficients[0][1:num_coefficients + 1] = first_half
  coefficients[0][num_coefficients + 1:] = symmetric_half

  return coefficients[0]


"""Generates a Fourier coefficient series that will result in a real signal
when the inverse transform is performed

The generation of the series uses the symmetric property of Fourier seires
and generates a Hermitian symmetric series.

X(w) = X(-w)*

Params:
  T - duration of signal
  dt - time step of signal
  limit - the maximum frequency of the Fourier coefficients
  seed - seed to use when generating random numbers

Returns:
  Fourier coefficient series
"""
def GetCoefficientsForRealSignal(T, dt, limit, seed=None):
  np.random.seed(seed)
  num_coefficients = int(T / dt / 2)
  first_half = np.random.normal(0, size=(1, num_coefficients)) \
      + 1j * np.random.normal(0, size=(1, num_coefficients))
  if limit:
    frequencies = 2.0 * np.pi / T / 2 / np.pi * (np.asarray(range(len(first_half[0]))) + 1)
    first_half[0][frequencies > limit] = 0

  symmetric_half = np.conj(first_half[0][::-1])
  coefficients = np.zeros(shape=(1, 2 * num_coefficients + 1), dtype=complex)
  # set dc to 0
  coefficients[0][0] = 0
  coefficients[0][1:num_coefficients + 1] = first_half
  coefficients[0][num_coefficients + 1:] = symmetric_half

  return coefficients[0]

"""Calculates the RMS of a series

Equation: sqrt( 1 / num_elements * sum(element^2))

Params:
  xs - a series

Returns:
  RMS value of the series
"""
def GetRMS(xs):
  return np.sqrt(np.average(xs**2))

"""
Creates a raster plot

Params:
  event_times_list - a list of event time
  color - color of vlines

Returns:
  ax - an axis containing the raster plot
"""
def raster(event_times_list, color='k'):
  import pylab
  ax = pylab.gca()
  for ith, trial in enumerate(event_times_list):
    pylab.vlines(trial, ith + .5, ith + 1.5, color=color)
  pylab.ylim(.5, len(event_times_list) + .5)

  return ax

"""
Get time array

Params:
  t - time length in seconds
  dt - time step size
  plus_one - add one extra time step

Returns:
  time array containing t / dt (+1) elements
"""
def GetTimeArray(t, dt, plus_one=False):
  if plus_one:
    t = t + dt

  return np.linspace(0, t, np.round(t / dt))
