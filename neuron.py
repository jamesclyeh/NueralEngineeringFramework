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

      return self.G(alpha * np.dot(np.array(e), np.array([xs])) + J_bias)

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
  """Leaky Integrate and Fire neuron model

  Equation:
    a = 1 / (tau_ref - tau_RC * ln(1 - 1 / J)
  """
  def __init__(self, tau_ref, tau_RC):
    self.tau_ref = tau_ref
    self.tau_RC = tau_RC

  """Get gain and bias given max firing rate and x-intercept

  0 = 1 / (tau_ref - tau_RC * ln(1 - 1 / (alpha * x_intercept + J_bias)))
  max_rate = 1 / (tau_ref - tau_RC * ln(1 - 1 / (alpha + J_bias)))

  tau_ref - tau_RC * ln(1 - 1 / (alpha + J_bias)) = 1 / max_rate
  tau_RC * ln(1 - 1 / (alpha + J_bias)) = tau_ref - 1 / max_rate
  ln(1 - 1 / (alpha + J_bias)) = (tau_ref - 1 / max_rate) / tau_RC
  1 - 1 / (alpha + J_bias) = e^((tau_ref - 1 / max_rate) / tau_RC)
  1 - e^((tau_ref - 1 / max_rate) / tau_RC) = 1 / (alpha + J_bias)
  alpha + J_bias = 1 / (1 - e^((tau_ref - 1 / max_rate) / tau_RC))
  J_bias = x - alpha
  0 = 1 / (tau_ref - tau_RC * ln(1 - 1 / (alpha * (x_intercept - 1) + x)))
  tau_ref - tau_RC * ln(1 - 1 / (alpha * (x_intercept - 1) + x)) = 1
  tau_ref - 1 = tau_RC * ln(1 - 1 / (alpha * (x_intercept - 1) + x))
  (tau_ref - 1) / tau_RC = ln(1 - 1 / (alpha * (x_intercept - 1) + x))
  1 / (1 - e^((tau_ref - 1) / tau_RC)) = alpha * (x_intercept - 1) + x
  1 / (1 - e^((tau_ref - 1) / tau_RC)) - x = alpha * (x_intercept - 1)
  alpha = (1 / (1 - e^((tau_ref - 1) / tau_RC)) - x )/ (x_intercept - 1)
  """
  def GetGainAndBias(self, max_firing_rate, x_intercept):
    x = 1. / (1 - np.exp((self.tau_ref - 1. / max_firing_rate) / self.tau_RC))
    alpha = (1. / (1 - np.exp((self.tau_ref - 1) / self.tau_RC)) - x )/ (x_intercept - 1)
    J_bias = x - alpha

    return alpha, J_bias


  def G(self, xs):
    ret = np.zeros(xs.shape)
    ret[xs > 1] = 1. / (self.tau_ref - self.tau_RC * np.log(1 - 1. / xs[xs > 1]))

    return ret
