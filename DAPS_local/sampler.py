import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE, get_k2_scheduler
from cores.mcmc import MCMCSampler
from forward_operator import LatentWrapper


def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        return LatentDAPS(**kwargs)
    return DAPS(**kwargs)


class DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config, k2_scheduler_config=None):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)
        self.k2_scheduler = None
        if k2_scheduler_config:
            # initialize adaptive k2 scheduler (skeleton)
            # strip non-constructor keys defensively
            try:
                _k2cfg = dict(k2_scheduler_config) if isinstance(k2_scheduler_config, dict) else k2_scheduler_config
                if hasattr(_k2cfg, 'pop'):
                    try:
                        _ = _k2cfg.pop('enabled')
                    except Exception:
                        pass
                else:
                    # create a plain dict without 'enabled' if present
                    _k2cfg = {k: v for k, v in (k2_scheduler_config.items() if isinstance(k2_scheduler_config, dict) else []) if k != 'enabled'}
                self.k2_scheduler = get_k2_scheduler(**_k2cfg)
            except Exception:
                # fallback: try raw
                self.k2_scheduler = get_k2_scheduler(**k2_scheduler_config)

        # Early-stop options
        self.early_stop_enabled = False
        self.early_stop_sigma = None
        # try reading from k2 config
        try:
            if k2_scheduler_config is not None:
                if hasattr(k2_scheduler_config, 'get'):
                    self.early_stop_enabled = bool(k2_scheduler_config.get('early_stop', False))
                    self.early_stop_sigma = float(k2_scheduler_config.get('min_sigma', 0.0)) if k2_scheduler_config.get('min_sigma', None) is not None else None
                elif isinstance(k2_scheduler_config, dict):
                    self.early_stop_enabled = bool(k2_scheduler_config.get('early_stop', False))
                    self.early_stop_sigma = float(k2_scheduler_config.get('min_sigma')) if k2_scheduler_config.get('min_sigma') is not None else None
        except Exception:
            pass
        # keep annealing sigma_min as fallback threshold
        try:
            if hasattr(annealing_scheduler_config, 'get'):
                self.annealing_sigma_min = float(annealing_scheduler_config.get('sigma_min', 0.0))
            elif isinstance(annealing_scheduler_config, dict):
                self.annealing_sigma_min = float(annealing_scheduler_config.get('sigma_min', 0.0))
            else:
                self.annealing_sigma_min = 0.0
        except Exception:
            self.annealing_sigma_min = 0.0
        # warm-up steps for K2 usage (default 10)
        try:
            if k2_scheduler_config:
                if hasattr(k2_scheduler_config, 'get'):
                    self.k2_warm_steps = int(k2_scheduler_config.get('warm_steps', 10))
                else:
                    self.k2_warm_steps = int(k2_scheduler_config.get('warm_steps', 10)) if isinstance(k2_scheduler_config, dict) else 10
            else:
                self.k2_warm_steps = 10
        except Exception:
            self.k2_warm_steps = 10

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method.

        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        T = self.annealing_scheduler.num_steps - 1
        pbar = tqdm.trange(T) if verbose else range(T)
        xt = x_start
        # 동적 sigma: 첫 스텝은 스케줄의 시작값으로 초기화하고 이후에는 K2 제안으로 갱신
        cur_sigma_val = float(self.annealing_scheduler.sigma_steps[0])
        used_steps = 0
        for step in pbar:
            # 현재 스텝에서 사용할 sigma는 항상 동적으로 유지되는 값
            sigma = cur_sigma_val
            # 선택: K2 이벤트 로거가 있으면 현재 사용 sigma를 기록해 디버깅 용이하게 함
            try:
                if getattr(self, 'k2_scheduler', None) is not None and getattr(self.k2_scheduler, 'logger', None) is not None:
                    self.k2_scheduler.logger.log({'mode': 'used', 'sigma_used': float(sigma)})
            except Exception:
                pass
            # 1. reverse diffusion
            with torch.no_grad():
                # Build scheduler safely: extract name, drop runtime-only keys like 'solver'
                cfg = self.diffusion_scheduler_config
                try:
                    name = cfg.get('name') if hasattr(cfg, 'get') else cfg['name']
                except Exception:
                    name = 'edm'
                other_kwargs = {}
                try:
                    for k in cfg:
                        if k in ('name', 'solver'):
                            continue
                        other_kwargs[k] = cfg[k]
                except Exception:
                    pass
                diffusion_scheduler = get_diffusion_scheduler(name=name, **other_kwargs, sigma_max=sigma)
                # Read solver for PF-ODE
                try:
                    solver = cfg.get('solver', 'euler') if hasattr(cfg, 'get') else ('euler')
                except Exception:
                    solver = 'euler'
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver=solver)
                x0hat = sampler.sample(xt)

            # 2. MCMC update
            x0y = self.mcmc_sampler.sample(xt, model, x0hat, operator, measurement, sigma, step / T)

            # 3. forward diffusion (마지막 스텝에서는 노이즈 주입 금지)
            if step < T - 1:
                # propose next sigma via K2 if available else use preset schedule
                if self.k2_scheduler is not None and step >= self.k2_warm_steps:
                    # Provide minimal required stats for K2 (CESS-only requires x0hat)
                    cur_sigma_t = torch.as_tensor(sigma, device=x0y.device, dtype=x0y.dtype).view(1, 1, 1, 1)
                    next_sigma, _ = self.k2_scheduler.propose_next(cur_sigma_t, x0y, stats={
                        'x0hat': x0hat,
                    })
                else:
                    next_sigma = self.annealing_scheduler.sigma_steps[step + 1]
                # Ensure device/dtype match and broadcastable shape
                next_sigma_t = torch.as_tensor(next_sigma, device=x0y.device, dtype=x0y.dtype)
                if next_sigma_t.ndim == 0:
                    next_sigma_t = next_sigma_t.view(1, 1, 1, 1)
                # Floor clamp: do not go slower than preset EDM schedule
                try:
                    sigma_floor = torch.as_tensor(self.annealing_scheduler.sigma_steps[step + 1], device=x0y.device, dtype=x0y.dtype).view(1, 1, 1, 1)
                    next_sigma_t = torch.minimum(next_sigma_t, sigma_floor)
                except Exception:
                    pass
                # Early stop: if next sigma reaches threshold
                next_sigma_scalar = float(next_sigma_t.flatten()[0].item())
                thresh = self.early_stop_sigma if (self.early_stop_sigma is not None) else self.annealing_sigma_min
                if bool(self.early_stop_enabled) and next_sigma_scalar <= (float(thresh) + 1e-8):
                    xt = x0y
                    used_steps += 1
                    self.used_anneal_steps = used_steps
                    break
                # otherwise proceed forward diffusion
                xt = x0y + torch.randn_like(x0y) * next_sigma_t
                # 다음 루프의 sigma를 next_sigma로 갱신 (동적 sigma)
                try:
                    cur_sigma_val = next_sigma_scalar
                except Exception:
                    cur_sigma_val = float(next_sigma_t.mean().item())
            else:
                xt = x0y

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    # Clamp and sanitize before evaluation to satisfy [0,1] after norm
                    x0hat_eval = torch.nan_to_num(x0hat, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                    x0y_eval = torch.nan_to_num(x0y, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                    x0hat_results = evaluator(gt, measurement, x0hat_eval)
                    x0y_results = evaluator(gt, measurement, x0y_eval)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            used_steps += 1
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        # store used steps if not early-stopped
        try:
            if getattr(self, 'used_anneal_steps', None) is None:
                self.used_anneal_steps = used_steps
        except Exception:
            pass
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        return x_start


class LatentDAPS(DAPS):
    """
    Latent Decoupled Annealing Posterior Sampling (LatentDAPS).

    Implements posterior sampling using a latent diffusion model combined with MCMC updates
    """
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using LatentDAPS in latent space, decoding intermediate results.

        Args:
            model (LatentDiffusionModel): Latent diffusion model.
            z_start (torch.Tensor): Initial latent state tensor.
            operator (nn.Module): Measurement operator applied in data space.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for monitoring performance.
            record (bool, optional): Whether to record intermediate states and metrics.
            verbose (bool, optional): Enables progress bar and evaluation metrics.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled data decoded from latent space.
        """
        if record:
            self.trajectory = Trajectory()
        T = self.annealing_scheduler.num_steps - 1
        pbar = tqdm.trange(T) if verbose else range(T)
        warpped_operator = LatentWrapper(operator, model)

        zt = z_start
        # 동적 sigma: 첫 스텝은 스케줄 시작값, 이후에는 제안으로 갱신
        cur_sigma_val = float(self.annealing_scheduler.sigma_steps[0])
        for step in pbar:
            sigma = cur_sigma_val
            # 1. reverse diffusion
            with torch.no_grad():
                cfg = self.diffusion_scheduler_config
                try:
                    name = cfg.get('name') if hasattr(cfg, 'get') else cfg['name']
                except Exception:
                    name = 'edm'
                other_kwargs = {}
                try:
                    for k in cfg:
                        if k in ('name', 'solver'):
                            continue
                        other_kwargs[k] = cfg[k]
                except Exception:
                    pass
                diffusion_scheduler = get_diffusion_scheduler(name=name, **other_kwargs, sigma_max=sigma)
                try:
                    solver = cfg.get('solver', 'euler') if hasattr(cfg, 'get') else ('euler')
                except Exception:
                    solver = 'euler'
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver=solver)
                z0hat = sampler.sample(zt)
                x0hat = model.decode(z0hat)

            # 2. MCMC update
            z0y = self.mcmc_sampler.sample(zt, model, z0hat, warpped_operator, measurement, sigma, step / T)
            with torch.no_grad():
                x0y = model.decode(z0y)

            # 3. forward diffusion (마지막 스텝에서는 노이즈 주입 금지)
            if step < T - 1:
                # Latent 경로도 동적 sigma 사용을 위해 next_sigma를 결정
                next_sigma = self.annealing_scheduler.sigma_steps[step + 1]
                next_sigma_t = torch.as_tensor(next_sigma, device=z0y.device, dtype=z0y.dtype)
                if next_sigma_t.ndim == 0:
                    next_sigma_t = next_sigma_t.view(1, 1, 1, 1)
                zt = z0y + torch.randn_like(z0y) * next_sigma_t
                try:
                    cur_sigma_val = float(next_sigma_t.flatten()[0].item())
                except Exception:
                    cur_sigma_val = float(next_sigma_t.mean().item())
            else:
                zt = z0y
            with torch.no_grad():
                xt = model.decode(zt)

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return xt

