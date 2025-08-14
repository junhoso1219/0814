#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}

cd /home/elicer/DAPS

ODE_STEPS=(5 10 20)
ANN_STEPS=(100 200 400)
ALGOS=(pixel pixel_mala pixel_hmc) # ULA, MALA, HMC

for ode in "${ODE_STEPS[@]}"; do
  for ann in "${ANN_STEPS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      TAG="pr_${algo}_ode${ode}_ann${ann}"
      echo "Running ${TAG}"
      python posterior_sample.py \
        +data=test-ffhq +model=ffhq256ddpm +task=phase_retrieval +sampler=edm_daps \
        task_group=${algo} save_dir=results num_runs=4 \
        sampler.diffusion_scheduler_config.num_steps=${ode} \
        sampler.annealing_scheduler_config.num_steps=${ann} \
        batch_size=100 name=${TAG} gpu=${GPU}
    done
  done
done

echo "Done: 100-sample sweep (ULA & HMC)"


