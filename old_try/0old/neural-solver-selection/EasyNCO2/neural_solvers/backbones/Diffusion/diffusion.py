import math
import torch
import numpy as np
import torch.nn.functional as F

class GaussianDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    self.betabar = np.cumprod(self.beta)
    self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
    self.alphabar = np.cumprod(self.alpha)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales
    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    return xt, epsilon

  def posterior(self, target_t, t, pred, xt, inference_trick = 'ddim'):
      if target_t is None:
          target_t = t - 1
      else:
          target_t = torch.from_numpy(target_t).view(1)

      atbar = self.alphabar[t]
      atbar_target = self.alphabar[target_t]

      if inference_trick is None or t <= 1:
          # Use DDPM posterior
          at = self.alpha[t]
          z = torch.randn_like(xt)
          atbar_prev = self.alphabar[t - 1]
          beta_tilde = self.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

          xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
          xt_target = xt_target + np.sqrt(beta_tilde).item() * z
      elif inference_trick == 'ddim':
          xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
          xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
      else:
          raise ValueError('Unknown inference trick {}'.format(inference_trick))

      return xt_target

########################################
# Discrete Diffusion
########################################

class CategoricalDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule, sparse):
    # Diffusion steps
    self.T = T
    self.sparse = sparse

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
    return torch.bernoulli(xt[..., 1].clamp(0, 1))

  def posterior(self, target_t, t, x0_pred_prob, xt, guided = False, grad = None):

      if target_t is None:
          target_t = t - 1
      else:
          target_t = torch.from_numpy(target_t).view(1)

      Q_t = np.linalg.inv(self.Q_bar[target_t]) @ self.Q_bar[t]
      Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
      Q_bar_t_source = torch.from_numpy(self.Q_bar[t]).float().to(x0_pred_prob.device)
      Q_bar_t_target = torch.from_numpy(self.Q_bar[target_t]).float().to(x0_pred_prob.device)

      if guided:
          xt_grad_zero = torch.zeros(xt.shape, device = xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
          xt_grad_one = torch.zeros(xt.shape, device = xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
          xt_grad_zero[..., 0] = (1 - xt) * grad
          xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
          xt_grad_one[..., 1] = xt * grad
          xt_grad_one[..., 0] = -xt_grad_one[..., 1]
          xt_grad = xt_grad_zero + xt_grad_one

      xt = F.one_hot(xt.long(), num_classes=2).float()
      xt = xt.reshape(x0_pred_prob.shape)

      x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
      x_t_target_prob_part_2 = Q_bar_t_target[0]
      x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

      x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

      sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
      x_t_target_prob_part_2_new = Q_bar_t_target[1]
      x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

      x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

      sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

      if guided:
          p_theta = torch.cat((1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)), dim = -1)
          p_phi = torch.exp(-xt_grad)
          if self.sparse:
              p_phi = p_phi.reshape(p_theta.shape)
          posterior = (p_theta * p_phi) / torch.sum((p_theta * p_phi), dim = 1, keepdim = True)

      if target_t > 0 and not guided:
          xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
      elif target_t <= 0 and not guided:
          xt = sum_x_t_target_prob.clamp(min=0)
      elif target_t > 0 and guided:
          xt = torch.bernoulli(posterior[..., 1].clamp(0, 1))
      elif target_t <= 0 and guided:
          xt = posterior[..., 1].clamp(min = 0)

      if self.sparse:
          xt = xt.reshape(-1)
      return xt, sum_x_t_target_prob.clamp(0, 1)

