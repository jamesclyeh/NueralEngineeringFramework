import numpy as np

class Neuron:
  """Get firing rates a(x) for neurons

  Equation:
    a_i = G_i[alpha_i * (x dot e_i) + J^bias_i]
  where:
    alpha - gain term, constrained to always be positive
    J^bias - constant bias term
    e - encoder, or preferred direction vector
    G - neuron model
    i - index of neuron

  Params:
  xs - a vector of values x being presented

  Returns:
  a(x) - a vector of a_i(x)
  """
  def GetFiringRates(self, xs, alpha, J_bias, e):
      return self.G(alpha * np.dot(np.array(e), np.array(xs).T) + J_bias)

  """Alternative GetGiringRates function which output size can be non
  standard.

  Note: the dot product of encoders and input has to be calculated prior
  to using this function.
  """
  def GetFiringRatesAlternate(self, xs, alpha, J_bias):
    return self.G(alpha * xs + J_bias)

  """Specific mathmetical model for neuron
  """
  def G(self, xs):
    raise NotImplementedError("Must implement G which is specific to \
      different neuron models")

class RectifiedLinearNeuron(Neuron):
  """RectifedLinear neuron model

  Equation:
    a = max(J, 0)
  """

  def G(self, xs):

    return np.maximum(0, xs)

  """Get gain and bias given max firing rate and x-intercept

  J = alpha * x + J_bias
  0 = alpha * x_intercept + J_bias
  x_intercept = -J_bias / alpha
  max_rate = alpha + J_bias

  J_bias = -x_intercept * alpha
  max_rate = alpha - x_intercept * alpha
  alpha(1 - x_intercept) = max_rate
  alpha = max_rate / (1 - x_intercept)
  """
  def GetGainAndBias(self, max_firing_rate, x_intercept):
    alpha = max_firing_rate / (1 - x_intercept)
    J_bias = -1 * x_intercept * alpha

    return alpha, J_bias

class LIFNeuron(Neuron):
  """Init for Leaky Integrate and Fire neuron model

  Params:
    tau_ref = refactory constant
    tau_RC = RC constant
  """
  def __init__(self, tau_ref, tau_RC):
    self.tau_ref = tau_ref
    self.tau_RC = tau_RC

  """Get gain and bias given max firing rate and x-intercept

  0 =  alpha * x_intercept + J_bias
  max_rate = 1 / (tau_ref - tau_RC * ln(1 - 1 / (alpha + J_bias)))

  tau_ref - tau_RC * ln(1 - 1 / (alpha + J_bias)) = 1 / max_rate
  tau_RC * ln(1 - 1 / (alpha + J_bias)) = tau_ref - 1 / max_rate
  ln(1 - 1 / (alpha + J_bias)) = (tau_ref - 1 / max_rate) / tau_RC
  1 - 1 / (alpha + J_bias) = e^((tau_ref - 1 / max_rate) / tau_RC)
  1 - e^((tau_ref - 1 / max_rate) / tau_RC) = 1 / (alpha + J_bias)
  alpha + J_bias = 1 / (1 - e^((tau_ref - 1 / max_rate) / tau_RC))
  J_bias = x - alpha

  alpha * x_intercept + x - alpha = 1
  alpha * (x_intercept - 1) = 1 - x
  alpha = (1 - x) / (x_intercept - 1)
  """
  def GetGainAndBias(self, max_firing_rate, x_intercept):
    x = 1. / (1. - np.exp((self.tau_ref - 1. / max_firing_rate) / self.tau_RC))
    alpha = (1. - x) / (x_intercept - 1.)
    J_bias = x - alpha

    return alpha, J_bias

  """Leaky Integrate and Fire neuron model

  Equation:
    a = 1 / (tau_ref - tau_RC * ln(1 - 1 / J) when J > 1
    a = 0 otherwise
  """
  def G(self, xs):
    ret = np.zeros(xs.shape)
    ret[xs > 1] = 1. / (self.tau_ref - self.tau_RC * np.log(1 - 1. / xs[xs > 1]))

    return ret

  def GetTemporalResponse(self, T, dt, J, V=0):
    num_points = J.size
    voltage = np.zeros(num_points)
    spikes = np.zeros(num_points)
    ref = -1
    voltage[0] = V
    for i in xrange(num_points - 1):
      if ref <= 0:
        dV = dt / self.tau_RC * (J[i] - voltage[i])
        voltage[i + 1] = min(1, max(0, voltage[i] + dV))
        if voltage[i+1] == 1:
          spikes[i + 1] = 1
          ref = self.tau_ref
      else:
        ref -= dt
    return voltage, spikes