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
    tau_ref - refactory constant
    tau_RC - RC constant
  """
  def __init__(self, tau_ref, tau_RC):
    self.tau_ref = tau_ref
    self.tau_RC = tau_RC

  """Get gain and bias given max firing rate and x-intercept

  Calculations:
    # Firing rate is 0 when J = 1
    1 =  alpha * x_intercept + J_bias
    # Max firing rate @ x boundary (x_max)
    max_rate = 1 / (tau_ref - tau_RC * ln(1 - 1 / (alpha * max_x + J_bias)))

    tau_ref - tau_RC * ln(1 - 1 / (alpha * max_x + J_bias)) = 1 / max_rate
    tau_RC * ln(1 - 1 / (alpha * max_x + J_bias)) = tau_ref - 1 / max_rate
    ln(1 - 1 / (alpha * max_x + J_bias)) = (tau_ref - 1 / max_rate) / tau_RC
    1 - 1 / (alpha * max_x + J_bias) = e^((tau_ref - 1 / max_rate) / tau_RC)
    1 - e^((tau_ref - 1 / max_rate) / tau_RC) = 1 / (alpha * max_x + J_bias)
    alpha * max_x + J_bias = 1 / (1 - e^((tau_ref - 1 / max_rate) / tau_RC))
    J_bias = x - alpha * max_x
    # sub ^ into J = 1
    alpha * x_intercept + x - alpha * max_x = 1
    alpha * (x_intercept - max_x) = 1 - x
    alpha = (1 - x) / (x_intercept - max_x)

  Params:
    max_firing_rate - maximum neuron firing rate
    x_intercept - x value at which firing rate is 0
    max_x - maximum value of possible x

  Returns:
    alpha (gain) and J_bias(bias) where J = alpha * x + J_bias
  """
  def GetGainAndBias(self, max_firing_rate, x_intercept, max_x=1.):
    x = 1. / (1. - np.exp((self.tau_ref - 1. / max_firing_rate) / self.tau_RC))
    alpha = (1. - x) / (x_intercept - max_x)
    J_bias = x - alpha * max_x

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

  """Calculate the temporal response of neuron using Euler's method

  Equation:
    dV/dt = 1 / tau_RC * [RJ_m - V]

  Params:
    T - length of time
    dt - time step
    J - input current
    V - initial voltage

  Returns:
    voltage, spikes
  """
  def GetTemporalResponse(self, T, dt, J, V=0):
    num_points = J.size
    voltage = np.zeros(num_points)
    spikes = np.zeros(num_points)
    ref = -1
    voltage[0] = V
    for i in xrange(num_points - 1):
      dV = dt / self.tau_RC * (J[i] - voltage[i])
      voltage[i+1] = max(0, voltage[i] + dV)
      ref -= dt
      voltage[i+1] *= min(1, max(0, (1 - ref / dt)))
      if voltage[i+1] > 1:
        overshoot = (voltage[i+1] - 1) / dV
        voltage[i+1] = 0
        spikes[i+1] = 1 / dt
        ref = self.tau_ref + dt * (1 - overshoot)
    return voltage, spikes
