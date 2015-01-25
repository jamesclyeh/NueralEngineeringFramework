import nengo
import numpy
import pylab
import numpy as np

n = nengo.neurons.LIFRate()

theta = numpy.linspace(0, 2*numpy.pi, 100)
x = numpy.array([numpy.cos(theta), numpy.sin(theta)])
pylab.plot(x[0],x[1])
pylab.axis('equal')

e = numpy.array([1.0, 1.0])
e = e/numpy.linalg.norm(e)

pylab.plot([0,e[0]], [0,e[1]],'r')

gain = 1
bias = 2.5

pylab.figure()
y = n.rates(numpy.dot(x.T, e), gain=gain, bias=bias)
pylab.plot(theta, y)
pylab.plot([numpy.arctan2(e[1],e[0])],0,'rv')
pylab.xlabel('angle')
pylab.ylabel('firing rate')
pylab.xlim(0, 2*numpy.pi)

from scipy.optimize import curve_fit
def func(x, a, b, c, d):
  return a * np.cos(b * x + c) + d
popt, pcov = curve_fit(func,theta, y)
pylab.plot(theta, popt[0] * np.cos(popt[1] * theta + popt[2]) + popt[3])
pylab.show()

