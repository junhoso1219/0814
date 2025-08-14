import tqdm
import torch
import numpy as np
import torch.nn as nn
from .trajectory import Trajectory


class MCMCSampler(nn.Module):
    """
    Monte Carlo sampler class for diffusion processes.

    Supports Langevin dynamics, Hamiltonian Monte Carlo (HMC) and Metropolis-Hastings (MH) methods.

    Attributes:
        num_steps (int): Number of sampling steps.
        lr (float): Initial learning rate.
        tau (float): Standard deviation for data-fitting term.
        lr_min_ratio (float): Minimum learning rate ratio.
        prior_solver (str): Method to compute prior score ('gaussian', 'exact', 'score-t', 'score-min').
        prior_sigma_min (float): Minimum sigma for prior computation.
        mc_algo (str): Monte Carlo algorithm ('langevin', 'mh', or 'hmc').
        momentum (float): Momentum coefficient for HMC.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01, prior_solver='gaussian', prior_sigma_min=1e-2,
                 mc_algo='langevin', momentum=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.prior_sigma_min = prior_sigma_min
        self.mc_algo = mc_algo
        self.momentum = momentum

    def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma):
        """
        Computes the conditional score function \nabla_x \log p(x_0 = x | x_t, y).

        Returns:
            Tuple containing:
                - Current score estimate.
                - Data-fitting loss.
        """
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2
        xt_term = (xt - x) / sigma ** 2
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)
        return data_term + xt_term + prior_term, data_fitting_loss

    def get_prior_score(self, x, x0hat, xt, model, sigma):
        if self.prior_solver == 'score-min' or self.prior_solver == 'score-t' or self.prior_solver == 'gaussian':
            prior_score = self.prior_score
        elif self.prior_solver == 'exact':
            prior_score = model.score(x, torch.tensor(self.prior_sigma_min).to(x.device)).detach()
        else:
            raise NotImplementedError
        return prior_score

    def prepare_prior_score(self, x0hat, xt, model, sigma):
        """
        Precomputes the prior score based on the specified solver method.
        """
        if self.prior_solver == 'score-min':
            self.prior_score = model.score(x0hat, self.prior_sigma_min).detach()

        elif self.prior_solver == 'score-t':
            self.prior_score = model.score(xt, sigma).detach()

        elif self.prior_solver == 'gaussian':
            self.prior_score = (x0hat - xt).detach() / sigma ** 2

        elif self.prior_solver == 'exact':
            pass

        else:
            raise NotImplementedError

    def mc_prepare(self, x0hat, xt, model, operator, measurement, sigma):
        """Prepares the sampler state before starting Monte Carlo sampling."""
        if self.mc_algo == 'hmc':
            self.velocity = torch.randn_like(x0hat)

    def mc_update(self, x, cur_score, lr, epsilon):
        """ Performs a single Monte Carlo update step (Langevin or HMC).

        Note: MALA is implemented inside `sample` to access target density and proposal asymmetry.
        """
        if self.mc_algo == 'langevin':
            x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
        elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
            step_size = np.sqrt(lr)
            self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon
            x_new = x + self.velocity * step_size
        else:
            raise NotImplementedError
        return x_new

    def sample_mh(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        if record:
            self.trajectory = Trajectory()
        
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for _ in pbar:
            # Gaussain proposal
            x_new = x + torch.randn_like(x) * np.sqrt(lr)
            # compute acceptance ratio
            loss_new = operator.loss(x_new, measurement)
            loss_old = operator.loss(x, measurement)
            log_data_ratio = (loss_old - loss_new) / (2 * self.tau ** 2)
            # compute prior p(x_0 | x_t)
            prior_loss_new = (x_new - x0hat).pow(2).flatten(1).sum(-1) 
            prior_loss_old = (x - x0hat).pow(2).flatten(1).sum(-1)
            log_prior_ratio = (prior_loss_old - prior_loss_new) / (2 * sigma ** 2)
            # compute acceptance probability
            log_accept_prob= log_data_ratio + log_prior_ratio
            accept = torch.rand_like(log_accept_prob).log() < log_accept_prob
            accept = accept.view(-1, *[1] * len(x.shape[1:]))
            # update: accept new sample
            x = torch.where(accept, x_new, x)
        return x.detach()

    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        """
        Main method for performing MCMC sampling.

        Args:
            xt (torch.Tensor): Current noisy latent tensor.
            model (nn.Module): Diffusion model providing the score function.
            x0hat (torch.Tensor): Initial estimate of x0 from PF-ODE.
            operator (Operator): Measurement operator.
            measurement (torch.Tensor): Measurement data.
            sigma (float): Noise scale at current timestep.
            ratio (float): Ratio to adjust learning rate scheduling.
            record (bool): Whether to record trajectory.
            verbose (bool): Verbosity flag.

        Returns:
            torch.Tensor: Sampled latent tensor.
        """
        if self.mc_algo == 'mh':
            return self.sample_mh(xt, model, x0hat, operator, measurement, sigma, ratio, record, verbose)
        if record:
            self.trajectory = Trajectory()
        lr = self.get_lr(ratio)
        self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
        self.prepare_prior_score(x0hat, xt, model, sigma)

        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        accepts = []
        for _ in pbar:
            cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
            epsilon = torch.randn_like(x)

            if self.mc_algo == 'mala':
                # Metropolis-Adjusted Langevin Algorithm
                step_size = lr
                mean_forward = x + step_size * cur_score
                x_prop = mean_forward + torch.sqrt(torch.tensor(2 * step_size, device=x.device, dtype=x.dtype)) * epsilon

                # log target density up to constant: data term + prior term
                def log_pi(z):
                    loss = operator.loss(z, measurement)
                    prior = (z - x0hat).pow(2).flatten(1).sum(-1)
                    return -(loss / (2 * self.tau ** 2) + prior / (2 * sigma ** 2))

                logp_x = log_pi(x).view(-1)
                logp_xp = log_pi(x_prop).view(-1)

                # reverse proposal mean uses score at x_prop
                prop_score, _ = self.score_fn(x_prop, x0hat, model, xt, operator, measurement, sigma)
                mean_reverse = x_prop + step_size * prop_score

                def log_q(z_to, mean_from):
                    # Gaussian with variance 2*step_size I
                    diff = (z_to - mean_from).flatten(1)
                    quad = (diff.pow(2).sum(-1)) / (4 * step_size)
                    return -quad

                log_q_x_given_xp = log_q(x, mean_reverse).view(-1)
                log_q_xp_given_x = log_q(x_prop, mean_forward).view(-1)
                log_accept_ratio = logp_xp - logp_x + log_q_x_given_xp - log_q_xp_given_x
                accept = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio
                accepts.append(accept.float().mean().detach().cpu().item())
                accept = accept.view(-1, *[1] * (x.dim() - 1))
                x = torch.where(accept, x_prop, x)
            else:
                x = self.mc_update(x, cur_score, lr, epsilon)

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x) 

            # record
            if record:
                self._record(x, epsilon, fitting_loss.sqrt())
        # optional: log acceptance rate summary if available via attached logger
        try:
            acc_rate = float(np.mean(accepts)) if accepts else None
            if hasattr(self, 'logger') and acc_rate is not None:
                self.logger.log({'mode': 'hmc', 'accept_rate': acc_rate})
        except Exception:
            pass
        return x.detach()

    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr

    def summary(self):
        print('+' * 50)
        print('MCMC Sampler Summary')
        print('+' * 50)
        print(f"Prior Solver    : {self.prior_solver}")
        print(f"MCMC Algorithm  : {self.mc_algo}")
        print(f"Num Steps       : {self.num_steps}")
        print(f"Learning Rate   : {self.lr}")
        print(f"Tau             : {self.tau}")
        print(f"LR Min Ratio    : {self.lr_min_ratio}")
        print('+' * 50)


class DAHMC(nn.Module):
    """
    Delayed-Acceptance HMC skeleton.

    Stage-1: cheap surrogate acceptance; Stage-2: exact acceptance.
    This is a placeholder to wire the API; full logic will be added iteratively.
    """
    def __init__(self, step_size_init: float = 0.01, leapfrog: int = 6, acc_target: float = 0.651,
                 surrogate: dict | None = None):
        super().__init__()
        self.step_size = step_size_init
        self.leapfrog = leapfrog
        self.acc_target = acc_target
        self.surrogate = surrogate or {}

    def forward(self, x, logpi_fn):
        return x
