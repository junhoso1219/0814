#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}

cd /home/elicer/DAPS

# 공통 하이퍼파라미터
COMMON_ARGS=(
  +data=demo-ffhq
  +model=ffhq256ddpm
  +task=phase_retrieval
  +sampler=edm_daps
  save_dir=results
  num_runs=4
  sampler.diffusion_scheduler_config.num_steps=5
  sampler.annealing_scheduler_config.num_steps=200
  batch_size=10
  data.start_id=0
  data.end_id=10
  gpu=${GPU}
)

# ULA (pixel)
python posterior_sample.py \
  "${COMMON_ARGS[@]}" \
  task_group=pixel \
  name=pr_ula

# MALA (pixel_mala)
python posterior_sample.py \
  "${COMMON_ARGS[@]}" \
  task_group=pixel_mala \
  name=pr_mala

# HMC (pixel_hmc)
python posterior_sample.py \
  "${COMMON_ARGS[@]}" \
  task_group=pixel_hmc \
  name=pr_hmc

echo "Done: PR demo ULA & HMC"


