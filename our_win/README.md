## our_win: DAPS ULA ↔ MALA ↔ HMC 실험 킷

이 폴더는 `DAPS` 레포(`/home/elicer/DAPS`) 위에서 ULA, MALA, HMC를 공정 비교(A/B)하고, 100장 스윕 및 결과 집계를 빠르게 실행하기 위한 스크립트를 제공합니다.

### 사전 준비
- DAPS 설치/체크포인트/데이터 다운로드는 레포 가이드대로 완료되어 있어야 합니다.
- 실행은 `DAPS` 루트에서 이루어지므로, 스크립트 내부에서 자동으로 `cd /home/elicer/DAPS` 합니다.

### 빠른 시작: PR 데모(10장) ULA ↔ MALA ↔ HMC
```bash
chmod +x /home/elicer/our_win/run_pr_demo.sh
/home/elicer/our_win/run_pr_demo.sh 0
```
- 인자: 첫 번째 인자는 GPU 인덱스(기본 0).
- 결과는 `DAPS/results/pr_ula_*`, `DAPS/results/pr_mala_*`, `DAPS/results/pr_hmc_*` 하위에 저장됩니다.

### 100장 스윕(FFHQ 픽셀 경로)
```bash
chmod +x /home/elicer/our_win/run_pr_100.sh
/home/elicer/our_win/run_pr_100.sh 0
```
- ODE 스텝 {5,10,20} × Anneal {100,200,400} × {ULA,MALA,HMC}를 순차 실행합니다.

### 결과 집계(CSV)
```bash
python /home/elicer/our_win/aggregate_results.py \
  --results_dir /home/elicer/DAPS/results \
  --output_csv /home/elicer/our_win/summary.csv
```
- 각 런 폴더의 `config.yaml`, `metrics.json`을 읽어 `name, task_group, model, data, ODE/anneal steps, PSNR/SSIM/LPIPS(mean/std)`를 정리합니다.

### 인터페이스 요약(중요)
- 레포는 `task_group`으로 MCMC 구성을 선택합니다.
  - `task=phase_retrieval`에서 픽셀 경로:
    - ULA: `task_group=pixel` (`mc_algo: langevin`)
    - MALA: `task_group=pixel_mala` (`mc_algo: mala`)
    - HMC: `task_group=pixel_hmc` (`mc_algo: hmc`)
- ODE/Anneal 스텝은 `sampler.diffusion_scheduler_config.num_steps`, `sampler.annealing_scheduler_config.num_steps`로 제어합니다.

참고: 레포의 `hmc`는 모멘텀 업데이트 기반이며 메트로폴리스 보정(accept/reject) 로깅은 기본 제공되지 않습니다. 따라서 acceptance 통계는 로그에 없고, 품질/지표 중심으로 비교합니다.


