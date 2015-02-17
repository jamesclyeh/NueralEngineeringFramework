import neuron
import simulator
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

  x = 0
  J = alpha * x + J_bias
  T = 1
  dt = 0.0001
  J = np.ones(T / dt) * J
  voltage, spikes = n.GetTemporalResponse(T, dt, J)
  t = np.linspace(0, T, T / dt)
  pylab.figure()
  pylab.vlines(t * spikes, 0.5, 1.5, color='r', linewidth=1)
  pylab.plot(t, voltage)
  print 'Simulated number of spikes: %s, expected 40' % sum(spikes)

  x = 1
  J = alpha * x + J_bias
  J = np.ones(T / dt) * J
  voltage, spikes = n.GetTemporalResponse(T, dt, J)
  pylab.figure()
  pylab.vlines(t * spikes, 0.5, 1.5, color='r', linewidth=1)
  pylab.plot(t, voltage)
  print 'Simulated number of spikes: %s, expected 150' % sum(spikes)

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
  pylab.xlabel('time (s)')
  pylab.xlim(0, 1)

  pylab.figure()
  pylab.plot(x, voltage)
  pylab.vlines(x * spikes, 0.5, 1.5, color='r', linewidth=1)
  pylab.xlabel('time (s)')
  pylab.xlim(0, 1)
  pylab.show()

def Question3():
  # Parameter validation
  tau_ref = 0.002
  tau_RC = 0.02
  J_bias = 1 / (1 - np.exp((tau_ref - 1 / 40.0) / tau_RC))
  alpha = 1 / (1 - np.exp((tau_ref - 1 / 150.0) / tau_RC)) - J_bias
  n = neuron.LIFNeuron(tau_ref, tau_RC)
  x = np.linspace(-1, 1, 100)
  pylab.plot(x, n.GetFiringRates(x, alpha=alpha, J_bias=J_bias, e=1))
  pylab.plot(x, n.GetFiringRates(x, alpha=alpha, J_bias=J_bias, e=-1))
  pylab.xlabel('J (current)')
  pylab.ylabel('$a$ (Hz)')

  # a
  x = 0
  T = 1
  dt = 0.001
  J1 = alpha * x + J_bias
  J2 = alpha * -1 * x + J_bias
  J1 = np.ones(T / dt) * J1
  J2 = np.ones(T / dt) * J2
  voltage1, spikes1 = n.GetTemporalResponse(T, dt, J1)
  voltage2, spikes2 = n.GetTemporalResponse(T, dt, J2)
  t = np.linspace(0, T, T / dt)

  pylab.figure()
  pylab.vlines(t * spikes1, 0.5, 1.5, color='r', linewidth=1)
  pylab.plot(t, np.zeros(len(t)))
  pylab.ylim(-0.5, 1.5)

  pylab.figure()
  pylab.vlines(t * spikes2, 0.5, 1.5, color='k', linewidth=1)
  pylab.plot(t, np.zeros(len(t)))
  pylab.ylim(-0.5, 1.5)

  # b
  x = 1
  J1 = alpha * x + J_bias
  J2 = alpha * -1 * x + J_bias
  J1 = np.ones(T / dt) * J1
  J2 = np.ones(T / dt) * J2
  voltage1, spikes1 = n.GetTemporalResponse(T, dt, J1)
  voltage2, spikes2 = n.GetTemporalResponse(T, dt, J2)

  pylab.figure()
  pylab.vlines(t * spikes1, 0.5, 1.5, color='r', linewidth=1)
  pylab.plot(t, np.zeros(len(t)))
  pylab.ylim(-0.5, 1.5)

  pylab.figure()
  pylab.vlines(t * spikes2, 0.5, 1.5, color='k', linewidth=1)
  pylab.plot(t, np.zeros(len(t)))
  pylab.ylim(-0.5, 1.5)

  # c
  t = np.linspace(0, T, T / dt)
  x = 0.5 * np.sin(10 * np.pi * t)
  J1 = alpha * x + J_bias
  J2 = alpha * -1 * x + J_bias
  voltage1, spikes1 = n.GetTemporalResponse(T, dt, J1)
  voltage2, spikes2 = n.GetTemporalResponse(T, dt, J2)

  pylab.figure()
  pylab.vlines(t * spikes1, 0.0, 1.0, color='r', linewidth=1)
  pylab.vlines(t * spikes2, -1.0, 0.0, color='k', linewidth=1)
  pylab.plot(t, x)
  pylab.ylim(-1.5, 1.5)

  pylab.show()

def Question4():
  # Note: question 3d is here too
  # 3d
  tau_ref = 0.002
  tau_RC = 0.02
  J_bias = 1 / (1 - np.exp((tau_ref - 1 / 40.0) / tau_RC))
  alpha = 1 / (1 - np.exp((tau_ref - 1 / 150.0) / tau_RC)) - J_bias
  n = neuron.LIFNeuron(tau_ref, tau_RC)
  T = 2.0
  dt = 0.001
  rms = 0.5
  max_frequency = 5
  t = np.linspace(0, T + dt, T / dt + 1)
  pylab.figure()
  pylab.title('Spikes generated with Gaussian white noise input')
  input_signal = utils.GetGaussianWhiteNoise(T, dt, rms, max_frequency)
  pylab.plot(t, input_signal)
  pylab.xlim(0, 2)
  J1 = alpha * input_signal + J_bias
  J2 = alpha * -1 * input_signal + J_bias
  voltage1, spikes1 = n.GetTemporalResponse(T, dt, J1)
  voltage2, spikes2 = n.GetTemporalResponse(T, dt, J2)
  pylab.vlines(t * spikes1, 0.0, 1.0, color='r', linewidth=1)
  pylab.vlines(t * spikes2, -1.0, 0.0, color='k', linewidth=1)

  # a
  # summing the two neuron, d1 = -d2 -> sum(neuron1, neuron2) = neuron1 - neuron2
  r = spikes1 - spikes2
  # convert into frequency domain to get rid of convolution
  R = np.fft.fftshift(np.fft.fft(r))
  # Using Gaussian window to improve decoding performance, by slicing the signal into
  # tiny time pieces and averaging over them window by window.
  freq = np.fft.fftshift(np.fft.fftfreq(len(R), d=dt))
  omega = freq * 2 * np.pi
  sigma_t = 0.025
  X = np.fft.fftshift(np.fft.fft(input_signal))
  W2 = np.exp(-omega ** 2 * sigma_t ** 2)
  W2 = W2 / sum(W2)
  # minimizing the square error
  # E = ( X(w) - X_hat(w) )^2
  # E = ( X(w) - R(w)H(w) )^2
  # 0 = X(w) - R(w)H(w)
  # H(w) = X(w)R*(w) / |R(w)|^2

  CP = X * R.conjugate()
  WCP = np.convolve(CP, W2, 'same')
  RP = R * R.conjugate()
  WRP = np.convolve(RP, W2, 'same')
  XP = X * X.conjugate()
  H = WCP / WRP
  # get the filter in time domain
  h = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H))).real
  # decoding using optimal decoder
  XHAT = H * R
  XHATP = XHAT * XHAT.conjugate()
  # get the decoded result in time domain
  xhat = np.fft.ifft(np.fft.ifftshift(XHAT)).real

  # b
  pylab.figure()
  pylab.subplot(1, 2, 1)
  pylab.plot(freq, H.real)   #
  pylab.xlabel('Hz')
  pylab.title('Optimal filter in frequency domain')
  pylab.xlim(-50, 50)

  pylab.subplot(1, 2, 2)
  pylab.plot(t-T/2, h)       #
  pylab.title('Optimal filter in time domain')
  pylab.xlabel('t')
  pylab.xlim(-0.5, 0.5)

  # c
  pylab.figure()
  pylab.plot(t, r, color='k', label='Sum of neuron responses', alpha=0.2)
  pylab.plot(t, input_signal, linewidth=2, label='Input signal')
  pylab.plot(t, xhat, label='Decoded input signal')
  pylab.title('$\hat{x}(t)$ vs. $x(t)$')
  pylab.legend(loc='best')
  pylab.xlim(0, t[-1])
  pylab.xlabel('t')

  # d
  pylab.figure()
  pylab.subplot(1, 2, 1)
  pylab.plot(freq, np.sqrt(XP), label='Power spectrum of X')
  pylab.legend()
  pylab.xlim(-8, 8)
  pylab.xlabel('$\omega$ (Hz)')
  pylab.ylabel('|X($\omega$)|')
  pylab.subplot(1, 2, 2)
  pylab.plot(freq, np.sqrt(RP), label='Spike response spectrum')
  pylab.legend()
  pylab.xlabel('$\omega$ (Hz)')
  pylab.ylabel('|R($\omega$)|')

  pylab.figure()
  pylab.plot(freq, np.sqrt(XHATP), label='Power spectrum of $\hat{X}$')
  pylab.legend()
  pylab.xlim(-15, 15)
  pylab.xlabel('$\omega$ (Hz)')
  pylab.ylabel('|$\hat{X}(\omega$)|')

  # 5c
  h2 = simulator.Simulator.GetPostSynapticFilter(t, 0.007)
  H2 = np.fft.fft(h2)
  f = np.fft.fftfreq(len(H2), dt)
  pylab.figure()
  pylab.subplot(1, 2, 1)
  pylab.plot(t - np.round(t[-1 * (len(t) % 2) + -1] / 2), h2)
  pylab.xlabel('t (seconds)')
  pylab.suptitle('Post synaptic current filter')
  pylab.xlim(-0.2, 0.2)
  pylab.subplot(1, 2, 2)
  pylab.scatter(f, (H2*H2.conjugate()).real)
  pylab.xlabel('$\omega$ (Hz)')

  fspikes1 = np.convolve(spikes1, h2, mode='same')
  fspikes2 = np.convolve(spikes2, h2, mode='same')
  A = np.array([fspikes1, fspikes2]).T
  decoders = simulator.Simulator.GetDecoders(input_signal, A)
  xhat2 = np.dot(A, decoders.T)
  pylab.figure()
  pylab.title('$\hat{x}(t)$ vs. $x(t)$')
  pylab.plot(t, input_signal, linewidth=2, label='Input signal')
  pylab.plot(t, xhat2, label='Decoded input signal')
  pylab.vlines(t * spikes1, 0.0, 1.0, color='r', linewidth=1)
  pylab.vlines(t * spikes2, -1.0, 0.0, color='k', linewidth=1)
  pylab.legend(loc='best')
  pylab.xlabel('t (seconds)')
  pylab.xlim(0.0, 2.0)

  # 5d
  input_signal = utils.GetGaussianWhiteNoise(T, dt, rms, 5)
  J1 = alpha * input_signal + J_bias
  J2 = alpha * -1 * input_signal + J_bias
  voltage1, spikes2_1 = n.GetTemporalResponse(T, dt, J1)
  voltage2, spikes2_2 = n.GetTemporalResponse(T, dt, J2)
  fspikes2_1 = np.convolve(spikes2_1, h2, mode='same')
  fspikes2_2 = np.convolve(spikes2_2, h2, mode='same')
  A2 = np.array([fspikes2_1, fspikes2_2]).T
  xhat3 = np.dot(A2, decoders.T)
  pylab.figure()
  pylab.title('$\hat{x}(t)$ vs. $x(t)$')
  pylab.plot(t, input_signal, linewidth=2, label='Input signal')
  pylab.plot(t, xhat3, label='Decoded input signal')
  pylab.legend(loc='best')
  pylab.xlabel('t (seconds)')
  pylab.xlim(0.0, 2.0)

  pylab.show()


def Question4E():
  # e
  T = 2.0
  dt = 0.001
  rms = 0.5
  t = utils.GetTimeArray(T, dt, True)
  tau_ref = 0.002
  tau_RC = 0.02
  J_bias = 1 / (1 - np.exp((tau_ref - 1 / 40.0) / tau_RC))
  alpha = 1 / (1 - np.exp((tau_ref - 1 / 150.0) / tau_RC)) - J_bias
  n = neuron.LIFNeuron(tau_ref, tau_RC)

  for freq in [2, 10, 30]:
    input_signal = utils.GetGaussianWhiteNoise(T, dt, rms, freq)
    J1 = alpha * input_signal + J_bias
    J2 = alpha * -1 * input_signal + J_bias
    voltage1, spikes1 = n.GetTemporalResponse(T, dt, J1)
    voltage2, spikes2 = n.GetTemporalResponse(T, dt, J2)
    h, _ = simulator.Simulator.GetTemporalOptimalDecoder(input_signal, spikes1 - spikes2, dt)
    pylab.plot(t, h)

  pylab.title('Optimal filters for signals of different frequencies')
  pylab.legend(['limit = 2', 'limit = 10', 'limit = 30'])
  pylab.xlim(0.9, 1.1)
  pylab.xlabel('t (seconds)')

  # f
  pylab.figure()
  for time in [1, 4, 10]:
    input_signal = utils.GetGaussianWhiteNoise(time, dt, rms, 5)
    freq = np.fft.fftshift(np.fft.fftfreq(len(input_signal), d=dt))
    J1 = alpha * input_signal + J_bias
    J2 = alpha * -1 * input_signal + J_bias
    voltage1, spikes1 = n.GetTemporalResponse(time, dt, J1)
    voltage2, spikes2 = n.GetTemporalResponse(time, dt, J2)
    h, H = simulator.Simulator.GetTemporalOptimalDecoder(input_signal, spikes1 - spikes2, dt)
    t = utils.GetTimeArray(time, dt, True)
    mid_point = len(h) / 2
    pylab.subplot(1, 2, 1)
    pylab.plot(t[0: 1 / dt] - 0.5, h[mid_point - 0.5 / dt: mid_point + 0.5 / dt])
    pylab.subplot(1, 2, 2)
    pylab.plot(freq, (H * H.conjugate()))

  pylab.suptitle('Optimal filters for signals of different length')
  pylab.subplot(1, 2, 1)
  pylab.xlabel('t (seconds)')
  pylab.xlim(-0.2, 0.2)
  pylab.subplot(1, 2, 2)
  pylab.legend(['T = 1', 'T = 4', 'T = 10'])
  pylab.xlabel('$\omega$ (Hz)')

  pylab.show()

def Question5():
  t = utils.GetTimeArray(0.1, 0.001)
  tau = 0.007
  h1 = simulator.Simulator.GetPostSynapticFilter(t, tau)
  h2 = simulator.Simulator.GetPostSynapticFilter(t, tau, 1)
  h3 = simulator.Simulator.GetPostSynapticFilter(t, tau, 2)
  h4 = simulator.Simulator.GetPostSynapticFilter(t, tau, 5)

  t = t - np.round(t[-1 * (len(t) % 2) + -1] / 2)
  pylab.plot(t, h1)
  pylab.plot(t, h2)
  pylab.plot(t, h3)
  pylab.plot(t, h4)
  pylab.title('Post synaptic current filter with varying n')
  pylab.xlabel('t (seconds)')
  pylab.legend(['n = 0', 'n = 1', 'n = 2', 'n = 5'])

  h1 = simulator.Simulator.GetPostSynapticFilter(t, 0.002)
  h2 = simulator.Simulator.GetPostSynapticFilter(t, 0.005)
  h3 = simulator.Simulator.GetPostSynapticFilter(t, 0.01)
  h4 = simulator.Simulator.GetPostSynapticFilter(t, 0.02)

  pylab.figure()
  pylab.plot(t, h1)
  pylab.plot(t, h2)
  pylab.plot(t, h3)
  pylab.plot(t, h4)
  pylab.title('Post synaptic current filter with varying tau')
  pylab.xlabel('t (seconds)')
  pylab.legend(['$\\tau$ = 0.002', '$\\tau$ = 0.005', '$\\tau$ = 0.01', '$\\tau$ = 0.02'])

  pylab.show()

if __name__ == "__main__":
  #Question1_1()
  #Question1_2()
  #Question2()
  #Question3()
  Question4()
  #Question4E()
  #Question5()