import neuron
import numpy as np
import pylab
import simulator as s

"""1.1
"""
def Question1():
  #a
  num_neurons = 16
  dx = 0.05
  x_high = 1
  x_low = -1
  x_in = np.array([np.linspace(x_low, x_high, (x_high - x_low) / dx)]).T
  sim = s.Simulator(
      neuron.RectifiedLinearNeuron(), x_in, [100, 200], [-0.95, 0.95], [-1, 1])
  x, responses = sim.GetNeuronResponses(num_neurons)
  pylab.xlabel('x (scalar)')
  pylab.ylabel('firing rate (Hz)')
  pylab.plot(x, responses)

  #b
  decoders = sim.GetDecoders(x, responses)
  xhat = np.dot(responses, decoders)

  #c
  pylab.figure()
  pylab.plot(x, x)
  pylab.plot(x, xhat)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}$')
  pylab.ylim(-1, 1)
  pylab.xlim(-1, 1)

  pylab.figure()
  pylab.plot(x, xhat-x)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}-x$')
  pylab.xlim(-1, 1)
  print 'RMSE no noise', np.sqrt(np.average((x-xhat)**2))

  #d
  noise_std = 0.2
  responses_noisy = responses + np.random.normal(
      scale=noise_std * np.max(responses), size=responses.shape)
  xhat_noisy = np.dot(responses_noisy, decoders)
  pylab.figure()
  pylab.xlabel('x (scalar)')
  pylab.ylabel('$firing rate_{noisy}$ (Hz)')
  pylab.plot(x, responses_noisy)
  pylab.figure()
  pylab.plot(x, x)
  pylab.plot(x, xhat_noisy)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}_{noisy}$')
  pylab.ylim(-1, 1)
  pylab.xlim(-1, 1)

  pylab.figure()
  pylab.plot(x, xhat_noisy-x)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}_{noisy}-x$')
  pylab.xlim(-1, 1)
  print 'RMSE noisy', np.sqrt(np.average((x-xhat_noisy)**2))

  #e
  decoders_noisy = sim.GetDecoders(x_in, responses, noise_std)
  xhat_with_decoder_noisy = np.dot(responses_noisy, decoders_noisy)
  pylab.figure()
  pylab.plot(x, x)
  pylab.plot(x, xhat_with_decoder_noisy)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}_{noisy} with decoder_{noisy}$')
  pylab.ylim(-1, 1)
  pylab.xlim(-1, 1)

  pylab.figure()
  pylab.plot(x, xhat_with_decoder_noisy-x)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}_{noisy} with decoder_{noisy}-x$')
  pylab.xlim(-1, 1)
  print 'RMSE with noise taken into account by decoder', \
    np.sqrt(np.average((x-xhat_with_decoder_noisy)**2))

  xhat_with_decoder_noisy2 = np.dot(responses, decoders_noisy)
  pylab.figure()
  pylab.plot(x, x)
  pylab.plot(x, xhat_with_decoder_noisy2)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}2_{noisy} with decoder_{noisy}$')
  pylab.ylim(-1, 1)
  pylab.xlim(-1, 1)

  pylab.figure()
  pylab.plot(x, xhat_with_decoder_noisy2-x)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}2_{noisy} with decoder_{noisy}-x$')
  pylab.xlim(-1, 1)
  print 'RMSE with noise taken into account by decoder when there is no noise', \
    np.sqrt(np.average((x-xhat_with_decoder_noisy2)**2))

  pylab.show()

"""1.2
"""
def Question2():
  #a
  dx = 0.05
  error_distortion = []
  error_noise = []
  noise_std = 0.01
  num_neurons = []
  dx = 0.05
  x_high = 1
  x_low = -1
  x_in = np.array([np.linspace(x_low, x_high, (x_high - x_low) / dx)]).T
  sim = s.Simulator(
      neuron.RectifiedLinearNeuron(), x_in, [100, 200], [-0.95, 0.95], [-1, 1])

  for i in range(10):
    num_neurons.append(2**(i+1))
    tmp_error_distortion = []
    tmp_error_noise = []
    for _ in range(10):
      x, responses = sim.GetNeuronResponses(2**(i+1))
      sigma = noise_std * np.max(responses)
      responses_noisy = responses + np.random.normal(
        scale=sigma, size=responses.shape)
      decoders = sim.GetDecoders(x, responses, noise_std)
      xhat = np.dot(responses_noisy, decoders)
      tmp_error_distortion.append(np.sum((x-xhat)**2)/2)
      tmp_error_noise = sigma ** 2 * sum(decoders ** 2)
    error_distortion.append(sum(tmp_error_distortion) / len(tmp_error_distortion))
    error_noise.append(sum(tmp_error_noise) / len(tmp_error_noise))

  pylab.figure()
  pylab.plot(num_neurons, error_distortion)
  offset = 5.5
  pylab.plot(
      num_neurons,
      [1.0 / n + offset / n for n in num_neurons], 'k--')
  pylab.plot(
      num_neurons,
      [1.0 / n**2 + offset / n**2 for n in num_neurons], 'k:')
  pylab.xlabel('N (number of neurons)')
  pylab.ylabel('$E_{distortion}$')
  pylab.gca().set_yscale('log',basex=10)
  pylab.gca().set_xscale('log',basex=10)
  pylab.legend(['neurons', '1/N', '$1/N^2$'])

  pylab.figure()
  pylab.plot(num_neurons, error_noise)
  offset = -0.999
  pylab.plot(
      num_neurons,
      [1.0 / n + offset / n for n in num_neurons], 'k--')
  pylab.plot(
      num_neurons,
      [1.0 / n**2 + offset / n**2 for n in num_neurons], 'k:')
  pylab.xlabel('N (number of neurons)')
  pylab.ylabel('$E_{noise}$')
  pylab.gca().set_yscale('log',basex=10)
  pylab.gca().set_xscale('log',basex=10)
  pylab.legend(['neurons', '1/N', '$1/N^2$'])

  pylab.show()

"""1.3
"""
def Question3():
  #a
  num_neurons = 16
  dx = 0.05
  x_high = 1
  x_low = -1
  x_in = np.array([np.linspace(x_low, x_high, (x_high - x_low) / dx)]).T
  sim = s.Simulator(
      neuron.LIFNeuron(0.002, 0.02), x_in, [100, 200], [-1, 1], [-1, 1])
  x, responses = sim.GetNeuronResponses(num_neurons)
  pylab.xlabel('x (scalar)')
  pylab.ylabel('firing rate (Hz)')
  pylab.plot(x, responses)

  #b
  noise_std = 0.2
  responses_noisy = responses + np.random.normal(
      scale=noise_std * np.max(responses), size=responses.shape)
  decoders_noisy = sim.GetDecoders(x, responses, noise_std)
  xhat_with_decoder_noisy = np.dot(responses_noisy, decoders_noisy)
  pylab.figure()
  pylab.plot(x, x)
  pylab.plot(x, xhat_with_decoder_noisy)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}_{noisy} with decoder_{noisy}$')
  pylab.ylim(-1, 1)
  pylab.xlim(-1, 1)

  pylab.figure()
  pylab.plot(x, xhat_with_decoder_noisy-x)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}_{noisy} with decoder_{noisy}-x$')
  pylab.xlim(-1, 1)
  print 'RMSE with noise taken into account by decoder', \
    np.sqrt(np.average((x-xhat_with_decoder_noisy)**2))

  xhat_with_decoder_noisy2 = np.dot(responses, decoders_noisy)
  pylab.figure()
  pylab.plot(x, x)
  pylab.plot(x, xhat_with_decoder_noisy2)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}2_{noisy} with decoder_{noisy}$')
  pylab.ylim(-1, 1)
  pylab.xlim(-1, 1)

  pylab.figure()
  pylab.plot(x, xhat_with_decoder_noisy2-x)
  pylab.xlabel('$x$')
  pylab.ylabel('$\hat{x}2_{noisy} with decoder_{noisy}-x$')
  pylab.xlim(-1, 1)
  print 'RMSE with noise taken into account by decoder when there is no noise', \
    np.sqrt(np.average((x-xhat_with_decoder_noisy2)**2))

  pylab.show()

"""2.1
"""
def Question4():
  #a
  n = neuron.LIFNeuron(0.002, 0.02)
  # -pi / 4
  e = np .array([1.0, -1.0])
  e = e / np.linalg.norm(e)

  a = np.linspace(-1,1,40)
  b = np.linspace(-1,1,40)

  X, Y = np.meshgrid(a, b)
  xs = zip(X.flatten(), Y.flatten())
  alpha, J_bias = n.GetGainAndBias(100, 0)

  from mpl_toolkits.mplot3d.axes3d import Axes3D
  fig = pylab.figure()
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  # n.GetFiringRates(xs, alpha, J_bias, e)
  p = ax.plot_surface(
      X, Y, n.GetFiringRatesAlternate(X * e[0] + Y * e[1], alpha, J_bias),
      linewidth=0, cstride=1, rstride=1, cmap=pylab.cm.jet)

  #b
  theta = np.linspace(-5 * np.pi / 4, np.pi * 3 / 4, 100).T
  x = np.array([np.cos(theta), np.sin(theta)]).T
  pylab.figure()
  pylab.plot(x[:,0],x[:,1])
  pylab.axis('equal')
  pylab.plot([0,e[0]], [0,e[1]],'r')

  pylab.figure()
  y = n.GetFiringRates(x, alpha, J_bias, e)
  pylab.plot(theta, y)
  pylab.plot([np.arctan2(e[1], e[0])], 0, 'rv')
  pylab.xlabel('Angle (radians)')
  pylab.ylabel('Firing Rate (Hz)')

  from scipy.optimize import curve_fit
  popt, pcov = curve_fit(
      lambda x, a, b, c, d: a * np.cos(b * x + c) + d,
      theta, y)
  pylab.plot(theta, popt[0] * np.cos(popt[1] * theta + popt[2]) + popt[3])

  pylab.show()

"""2.2
"""
def Question5():
  #a
  encoders = np.apply_along_axis(
      lambda e:  e / np.linalg.norm(e), 1,
      np.random.uniform(-1, 1, (100, 2)))

  theta = np.linspace(0, 2 * np.pi, 100)
  circle = np.array([np.cos(theta), np.sin(theta)])
  pylab.figure()
  pylab.title('Encoders')
  pylab.plot(circle[0], circle[1])
  pylab.axis('equal')
  pylab.plot([np.zeros(encoders[:, 0].shape), encoders[:, 0]],
      [np.zeros(encoders[:, 1].shape), encoders[:, 1]],'r')

  #b
  num_neurons = 100
  a = np.linspace(-1,1,40)
  b = np.linspace(-1,1,40)

  X, Y = np.meshgrid(a, b)
  xs = np.array(zip(X.flatten(), Y.flatten()))

  sim = s.Simulator(
      neuron.LIFNeuron(0.002, 0.02), xs, [100, 200], [-1, 1],
      encoders = encoders)
  x, responses = sim.GetNeuronResponses(num_neurons)
  noise_std = 0.2
  responses_noisy = responses + np.random.normal(
      scale=noise_std * np.max(responses), size=responses.shape)
  decoders_noisy = sim.GetDecoders(x, responses, noise_std)

  pylab.figure()
  pylab.title('Decoders')
  pylab.axis('equal')
  pylab.plot([np.zeros(decoders_noisy[:, 0].shape), decoders_noisy[:, 0]],
      [np.zeros(decoders_noisy[:, 1].shape), decoders_noisy[:, 1]],'r')

  decoder_normalized = np.apply_along_axis(
      lambda e:  e / np.linalg.norm(e), 1, decoders_noisy)

  pylab.figure()
  pylab.title('Decoders')
  pylab.axis('equal')
  pylab.plot([np.zeros(decoder_normalized[:, 0].shape), decoder_normalized[:, 0]],
      [np.zeros(decoder_normalized[:, 1].shape), decoder_normalized[:, 1]],'r')

  #c
  x_rand = np.random.uniform(-0.8, 0.8, (20, 2))
  sim.SetInput(x_rand)
  x, responses = sim.GetNeuronResponses(num_neurons)
  xhat = np.dot(responses, decoders_noisy)

  pylab.figure()
  pylab.title('Decoding using calculated decoders')
  pylab.axis('equal')
  pylab.scatter(x_rand[:, 0], x_rand[:, 1])
  pylab.scatter(xhat[:, 0], xhat[:, 1], c='r')
  pylab.plot(circle[0], circle[1])
  pylab.legend(['unit circle', 'input', 'decoded value'])
  print 'RMSE decoding using calculated decoder', \
    np.sqrt(np.average((x_rand-xhat)**2))

  #d
  encoder_normalized = np.zeros(encoders.shape)
  for i in xrange(len(encoders)):
    encoder_normalized[i] = encoders[i] * np.linalg.norm(decoders_noisy[i])
  xhat2 = np.dot(responses, encoders)
  xhat3 = np.dot(responses, encoder_normalized)
  pylab.figure()
  pylab.title('Decoding using encoders')
  pylab.axis('equal')
  pylab.scatter(x_rand[:, 0], x_rand[:, 1])
  pylab.scatter(xhat2[:, 0], xhat2[:, 1], c='r')
  pylab.plot(circle[0], circle[1])
  pylab.legend(['unit circle', 'input', 'decoded value'])
  print 'RMSE decoding using encoder', np.sqrt(np.average((x_rand-xhat2)**2))
  pylab.figure()
  pylab.title('Decoding using normalized encoders')
  pylab.axis('equal')
  pylab.scatter(x_rand[:, 0], x_rand[:, 1])
  pylab.scatter(xhat3[:, 0], xhat3[:, 1], c='r')
  pylab.plot(circle[0], circle[1])
  pylab.legend(['unit circle', 'input', 'decoded value'])
  print 'RMSE decoding using normalized encoder', \
      np.sqrt(np.average((x_rand-xhat3)**2))

  pylab.show()

if __name__ == "__main__":
  #Question1()
  #Question2()
  #Question3()
  #Question4()
  Question5()
