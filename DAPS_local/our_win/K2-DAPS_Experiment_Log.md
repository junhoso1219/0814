## K2‑DAPS 실험 기록서 (공정 비교 · 문제 해결 · 코드 변경 내역)

### 0) 목적과 합격선(비열등 기준)
- 목적: DAPS에서 ULA → HMC/MALA로 치환 시 “편향 제거 + 성능/안정성”을 정량 검증하고, K2‑DAPS(CESS 적응·비단조 재가열·DA‑HMC)를 통해 공정 조건에서 동품질·시간절감 달성
- 공정 비교 기본 조건(픽셀 HMC): `task_group=pixel_hmc`, ODE=5, ANN=200, solver=euler, batch=10, data 0–100, num_runs=4
- 비열등 합격선: ΔPSNR ≥ −0.10 dB, ΔLPIPS ≤ +0.005, ΔSSIM ≥ −0.002, 그리고 시간 20%↓ 또는 유효 어닐링 스텝 25%↓

### 1) 환경·데이터·베이스라인
- 환경: Torch 2.4.1+cu121, CUDA 12.1, GPU: NVIDIA A100 80GB PCIe MIG(2g.20gb)
- 데이터/태스크: FFHQ‑256, Phase Retrieval(`task.operator.sigma=0.05`)
- 베이스라인: HMC, Euler(ODE=5), ANN=200, run‑avg 평가(논문 30dB대는 보통 ODE=10/ANN=400 + best‑of 조건)

### 2) 코드 변경 내역(파일별)

#### 2.1 `DAPS_local/sampler.py`
- 동적 σ 전파(핵심)
  - 루프 외부에 `cur_sigma_val` 도입 → 역확산/우도/전방 모두 현재 σ 사용
  - 다음 스텝 σ는 K2의 `next_sigma`로 갱신
- 마지막 스텝 처리(off‑by‑one 수정)
  - `T = self.annealing_scheduler.num_steps - 1`
  - 루프 `for step in range(T)`·전방 노이즈 `if step < T - 1`에서만 주입 → 마지막에는 노이즈 재주입 없음
- K2 호출·배선
  - `cur_sigma_t`를 `[1,1,1,1]` 텐서로 캐스팅 후 `propose_next` 호출
  - `next_sigma`는 `as_tensor(...).view(1,1,1,1)`로 브로드캐스트/디바이스·dtype 정합 보장
  - 이벤트 로깅: `{'mode':'used','sigma_used':...}` 추가로 σ 전파 디버깅 용이
- ODE 솔버 설정 분리
  - EDM 스케줄러 생성 시 `name`과 runtime 키 분리(EDMScheduler는 `solver` 미지원)
  - PF‑ODE에만 `solver`(`euler` 기본, 필요 시 `rk4` 등) 전달
- 평가 안정 가드
  - `nan_to_num` + `clamp(-1,1)` 후 evaluator 호출 → PIQ 지표 안전
- MCMC 진행도 정규화
  - `step / self.annealing_scheduler.num_steps` → `step / T`로 정합
- K2 warm‑up(초기 불안정 완화)
  - `self.k2_warm_steps`(기본 10) 이전에는 preset 스케줄 사용, 이후 K2 활성화
- Floor 클램프(베이스라인 이하로 느려지지 않게)
  - `sigma_floor = annealing_scheduler.sigma_steps[step+1]`
  - `next_sigma_t = min(next_sigma_t, sigma_floor)`

#### 2.2 `DAPS_local/cores/scheduler.py`
- `K2Scheduler`(CESS 기반 적응)
  - CESS 이진탐색(Δγ), log‑sum‑exp 안정화, Δγ 상한(`max_step_factor`) 적용
  - EMA 스무딩(`ema_alpha`), 배치 스칼라화(`median`), 디바이스/ dtype 정합
  - 단조 감소 클램프(`torch.minimum(sigma_out, cur_scalar)`)
  - 이벤트 로깅(JSONL) 지원(`{'mode':'cess','sigma','next_sigma','dgamma','cess',...}`)
  - 반환: `[1,1,1,1]` 텐서(브로드캐스트 안전)
- r 정규화(안정성 향상)
  - `sum` → `mean`으로 스케일 정규화
  - 추가로 `r /= cur_sigma^2 + 1e-8`로 σ 스케일 보정

#### 2.3 `DAPS_local/posterior_sample.py`
- `JSONLLogger` 주입: `results/<name>/events.jsonl` 생성
- 종료 시 `timing.json` 기록: `total_wall_sec`, `per_image_sec`, `ode_steps`, `anneal_steps_cfg`, `used_anneal_steps(설정 기반)`
- 저장 루프 IndexError 수정: 마지막 미니배치 안전(`for idx in range(len(pil_image_list))`)

#### 2.4 `DAPS_local/utils/jsonl_logger.py`
- 라인 버퍼 JSONL 로거(단순) 추가

### 3) 주요 버그/이슈와 해결
- GPU/CPU 미스매치: `next_sigma` 디바이스/ dtype 정합·브로드캐스트 `[1,1,1,1]`로 해결
- 마지막 스텝 노이즈 재주입: 오프바이원 수정(`T=...`, `step < T-1`)으로 해결
- K2 σ 역전(증가) 현상: 단조 클램프 도입으로 해결
- K2 σ 스칼라화: `median().view(1,1,1,1)`로 고정, 다운스트림 일관성 확보
- EDM 스케줄러에 `solver` 전달 오류: 이름·런타임 키 분리, PF‑ODE에만 `solver` 적용
- Hydra override 문법
  - 기존 키 덮어쓰기: 접두사 없이(`task.pixel_hmc.mcmc_sampler_config.lr_min_ratio=0.5`)
  - 새 키 추가·강제: `+`/`++` 사용
- DictConfig `pop` 에러: `_check()`에서 `pop('solver')` 제거, 런타임 분리로 대체
- 평가 NaN/범위 에러: `nan_to_num` + `clamp` 선처리
- 타이밍 파일 미생성: `KeyboardInterrupt`로 중단된 런 → 정상 종료 시 생성됨
- 속도 차이: `rk4`는 디버깅용(느림), 공정 비교는 `euler` 고정

### 4) K2‑CESS 하이퍼와 공정 비교 전략
- 고정: `task_group=pixel_hmc`, Euler(ODE=5), ANN=200, batch=10, data 0–100, runs=4, `min_sigma=0.05`
- 스윕(스모크 20장, run=1): `target_cess∈{0.85,0.90,0.95}`, `max_step_factor∈{2,3,5}`, `ema_alpha∈{0.2,0.3}`
- 선별 기준: PSNR↑, 그리고 used 스텝↓ 또는 per_image_sec↓
- 본런: 상위 2–3개 조합으로 100장·run=4 재측정

### 5) 유용한 커맨드(요약)
- HMC 기준선(Euler/5·200)
```
python posterior_sample.py ... task_group=pixel_hmc \
  +sampler.diffusion_scheduler_config.solver=euler \
  sampler.diffusion_scheduler_config.num_steps=5 \
  sampler.annealing_scheduler_config.num_steps=200 \
  sampler.k2_scheduler_config=null
```
- K2‑CESS(공정 비교, 예: R=0.90, F=3, A=0.2)
```
python posterior_sample.py ... task_group=pixel_hmc \
  +sampler.diffusion_scheduler_config.solver=euler \
  sampler.diffusion_scheduler_config.num_steps=5 \
  sampler.annealing_scheduler_config.num_steps=200 \
  ++sampler.k2_scheduler_config.name=k2 \
  ++sampler.k2_scheduler_config.target_cess=0.90 \
  ++sampler.k2_scheduler_config.max_step_factor=3.0 \
  ++sampler.k2_scheduler_config.ema_alpha=0.2 \
  ++sampler.k2_scheduler_config.min_sigma=0.05
```
- 스윕(스모크 20장)
```
for R in 0.85 0.90 0.95; do
  for F in 2 3 5; do
    for A in 0.2 0.3; do
      python posterior_sample.py ... data.start_id=0 data.end_id=20 num_runs=1 \
        ++sampler.k2_scheduler_config.target_cess=$R \
        ++sampler.k2_scheduler_config.max_step_factor=$F \
        ++sampler.k2_scheduler_config.ema_alpha=$A \
        ++sampler.k2_scheduler_config.min_sigma=0.05 \
        +eval_fn_list='[psnr]' \
        name=pr_k2_grid_R${R/./}_F${F}_A${A/./}_ode5_ann200_quick
    done
  done
done
```

### 6) 진단 방법(빠른 체크)
- σ 전파·단조·말단 도달
```
head -n 6 results/<run>/events.jsonl
tail -n 10 results/<run>/events.jsonl
```
- 자동 검사 스니펫
```
python - <<'PY'
import json, pathlib
p=pathlib.Path('results/<run>/events.jsonl')
prop=mono=True; prev=None; last=None
for ln in p.open():
    ev=json.loads(ln)
    if ev.get('mode')=='cess':
        if prev is not None: prop &= abs(prev - float(ev['sigma']))<1e-6
        mono &= float(ev['next_sigma']) <= float(ev['sigma'])+1e-9
        last=float(ev['next_sigma']); prev=last
print('propagates=',prop,'monotone=',mono,'last_next_sigma=',last)
PY
```
- 유효 스텝(`"mode":"used"` 카운트), 시간(`timing.json`), 품질(`metrics.json`)

### 7) 중간 결과와 해석(요약)
- Euler/5·200, run‑avg 기준 PSNR 18–20 dB는 비정상 아님(논문 30 dB대는 ODE=10/ANN=400 + best‑of)
- tail(저σ) 구간 불안정 시 중간(σ 중간)에서 18–19 dB 찍히다가 최종 14–16 dB로 떨어짐 → HMC 스텝 크기/리프로그 과함, CESS 지나친 하강, ODE 스텝 거침 등이 원인
- 공정 조건 유지하에 K2‑CESS 튜닝만으로 비열등 + 효율 달성 가능(스윕에서 상위 조합 선별 후 본런)

### 8) 다음 단계(권장)
- HMC 수락률 로깅 추가(튜닝 용이)
- K2 Phase‑3: KSD(IMQ) + 재가열(plateau/낮은 수락률/낮은 CESS 동시 트리거)
- Phase‑4: DA‑HMC(서로게이트→정밀 2단 수락)로 비용↓
- 결과 집계 스크립트(PSNR/LPIPS/SSIM/시간/used 스텝 테이블) 정리

### 9) FAQ / 팁
- Hydra: 기존 키 덮어쓰기는 접두사 없이, 새 키 추가는 `+`/강제는 `++`
- EDM: `solver`는 스케줄러에 전달하지 않음(오류), PF‑ODE에만 전달
- 디바이스: σ/샘플 텐서 디바이스/ dtype·브로드캐스트 정합 필수
- 공정성: 비교는 항상 `task_group=pixel_hmc`, Euler, ODE/ANN 동일

---
본 문서는 실험 전 과정을 재현할 수 있도록 코드 변경, 발생 이슈, 커맨드, 진단, 해석을 한 곳에 정리한 기록입니다. 필요한 보완(예: 수락률 로깅, 결과 테이블 자동화)은 후속 커밋으로 추가 가능합니다.



### 10) 금일 스모크(A/B) 실행 요약 및 해석
- 실행 요약(Euler 고정, 0–20 슬라이스, num_runs=1)
  - K2‑CESS 스모크: `target_cess=0.90, max_step_factor=3.0, ema_alpha=0.2, min_sigma=0.10`
    - 결과: 평균 PSNR ≈ 17.636, 벽시계 ≈ 802.8s, per‑image PSNR 분산 큼(≈ 8–31 dB)
    - 비고: 관측 노이즈가 0.05인데 `min_sigma=0.10`으로 보수적 → 테일 디테일 일부 손실 가능
  - HMC 베이스라인(Euler) A/B 시도: `enabled=false`만 전달 시 K2 생성 경로에서 오류 발생(아래 11) 참조)

- 배치 간 진행 PSNR(진행 막대) 20 dB vs 14 dB 차이 원인
  - 두 런 모두 ODE 적분기는 Euler였음(설정 확인). 차이는 “진행 중 미니배치 평균” 때문.
  - PR은 이미지 난이도 분산이 커서 배치 구성에 따라 진행 PSNR 평균이 5–8 dB 이상 차이 가능.
  - 대표성 판단은 최종 집계(전체 100장, run≥4, best‑of 옵션 시 더 상승) 기준으로 수행.

### 11) 오류 스냅샷과 핫픽스(K2 비활성화 관련)
- 증상 A: `get_k2_scheduler() missing 1 required positional argument: 'name'`
  - 원인: K2를 끄려 `enabled=false`만 전달했는데, 생성 시 `name`이 필요함.
  - 임시 해법: 이름은 넣어주기 → `++sampler.k2_scheduler_config.name=k2`.

- 증상 B: `TypeError: K2Scheduler.__init__() got an unexpected keyword argument 'enabled'`
  - 원인: `enabled` 키가 그대로 생성자에 전달됨.
  - 즉시 우회(명령): K2를 완전 off 대신 no‑op로 만들어 베이스라인과 동일 동작 유도
    ```bash
    ++sampler.k2_scheduler_config.name=k2 \
    ++sampler.k2_scheduler_config.target_cess=1.0 \
    ++sampler.k2_scheduler_config.max_step_factor=1.0 \
    ++sampler.k2_scheduler_config.ema_alpha=0.0 \
    ++sampler.k2_scheduler_config.min_sigma=0.05
    ```
  - 근본 조치(코드):
    - `sampler.py`에서 K2 생성 가드: `enabled`가 없거나 false면 생성하지 않음(생성 전 `enabled` pop).
    - `K2Scheduler.__init__`에 `**kwargs` 수용해 불필요 키 무시.

### 12) 공정 비교 A/B 커맨드(슬라이스·동일조건)
- 동일 슬라이스(예: 0–20), Euler 고정, 배치 10, ODE=5, ANN=200
- HMC 베이스라인(Euler, no‑op K2)
  ```bash
  /usr/bin/time -f "wall_sec=%e maxrss=%MKB" \
  python posterior_sample.py \
    +data=test-ffhq +model=ffhq256ddpm +task=phase_retrieval +sampler=edm_daps \
    task_group=pixel_hmc save_dir=results num_runs=1 \
    +sampler.diffusion_scheduler_config.solver=euler \
    sampler.diffusion_scheduler_config.num_steps=5 \
    sampler.annealing_scheduler_config.num_steps=200 \
    ++sampler.k2_scheduler_config.name=k2 \
    ++sampler.k2_scheduler_config.target_cess=1.0 \
    ++sampler.k2_scheduler_config.max_step_factor=1.0 \
    ++sampler.k2_scheduler_config.ema_alpha=0.0 \
    ++sampler.k2_scheduler_config.min_sigma=0.05 \
    save_traj=false batch_size=10 data.start_id=0 data.end_id=20 \
    name=ab_hmc_euler_baseline_s0_20_noopk2 gpu=0 \
  | tee /home/elicer/our_win/ab_hmc_euler_baseline_s0_20_noopk2.txt
  ```
- K2‑CESS 비교(Euler, 공정화: `min_sigma=0.05`)
  ```bash
  /usr/bin/time -f "wall_sec=%e maxrss=%MKB" \
  python posterior_sample.py \
    +data=test-ffhq +model=ffhq256ddpm +task=phase_retrieval +sampler=edm_daps \
    task_group=pixel_hmc save_dir=results num_runs=1 \
    +sampler.diffusion_scheduler_config.solver=euler \
    sampler.diffusion_scheduler_config.num_steps=5 \
    sampler.annealing_scheduler_config.num_steps=200 \
    ++sampler.k2_scheduler_config.name=k2 \
    ++sampler.k2_scheduler_config.target_cess=0.90 \
    ++sampler.k2_scheduler_config.max_step_factor=3.0 \
    ++sampler.k2_scheduler_config.ema_alpha=0.2 \
    ++sampler.k2_scheduler_config.min_sigma=0.05 \
    save_traj=false batch_size=10 data.start_id=0 data.end_id=20 \
    name=ab_k2_cess_euler_min005_s0_20 gpu=0 \
  | tee /home/elicer/our_win/ab_k2_cess_euler_min005_s0_20.txt
  ```

### 13) 권장 운영 가이드(현 목표: 동품질‑시간단축)
- 적분기: Euler 고정(레포 기본, 시간 효율 우수)
- 관측 노이즈 0.05 → `min_sigma=0.05`로 공정 매칭
- HMC 수락률 목표: 0.6–0.7 (낮으면 `lr`↓ 또는 `lr_min_ratio`↑로 테일 안정화)
- K2‑CESS 튠 프로파일
  - quality‑first: `target_cess=0.95`, `max_step_factor=2.0`, `ema_alpha∈{0.2,0.3}`
  - speed‑first: `target_cess=0.90`, `max_step_factor=3.0`, `ema_alpha=0.2`

### 14) 조기종료(early-stop) 구현 · 확인 · 주의사항
- 구현(코드 반영 완료)
  - `sampler.DAPS.sample` 안에서 K2 제안 `next_sigma`가 임계치(기본: `k2.min_sigma` 또는 EDM `sigma_min`) 이하가 되면 `break`.
  - 실제 사용 스텝을 `sampler.used_anneal_steps`에 저장. `posterior_sample.py`의 `timing.json`은 이를 우선 기록.
  - K2 생성 시 `enabled` 등 불필요 키는 무시하도록 방어, `K2Scheduler.__init__`는 `**kwargs` 허용.
- 실행 예(PSNR‑only, 0–20, Euler, runs=1)
  - K2(speed): `name=time_ab_k2_psnr_early`
  ```bash
  source /home/elicer/.venvs/daps/bin/activate; cd /home/elicer/DAPS_local; /usr/bin/time -f "wall_sec=%e maxrss=%MKB" python posterior_sample.py +data=test-ffhq +model=ffhq256ddpm +task=phase_retrieval +sampler=edm_daps task_group=pixel_hmc save_dir=results num_runs=1 +sampler.diffusion_scheduler_config.solver=euler sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 ++sampler.k2_scheduler_config.name=k2 ++sampler.k2_scheduler_config.target_cess=0.90 ++sampler.k2_scheduler_config.max_step_factor=3.0 ++sampler.k2_scheduler_config.ema_alpha=0.2 ++sampler.k2_scheduler_config.min_sigma=0.05002 ++sampler.k2_scheduler_config.early_stop=true +eval_fn_list='[psnr]' save_traj=false batch_size=10 data.start_id=0 data.end_id=20 name=time_ab_k2_psnr_early gpu=0 | tee /home/elicer/our_win/time_ab_k2_psnr_early.txt
  ```
- 확인 방법
  - `results/<run>/timing.json` → `used_anneal_steps < 200`면 조기종료 적용.
  - `events.jsonl` tail에 `sigma_used`, `next_sigma`가 임계 부근에서 고정.
- 주의: 부동소수 오차로 임계 통과가 안 될 수 있으니 `min_sigma`를 소폭 상향(예: 0.05002) 권장.

### 15) 공정 시간 비교(PSNR‑only) 결과 요약(0–20, Euler, runs=1)
- HMC(no‑op K2): `time_ab_hmc_psnr`
  - PSNR mean 17.696, wall 811.08s, `used_anneal_steps=200`
- K2‑CESS(speed, tc=0.90,f=3,ema=0.2, minσ=0.05): `time_ab_k2_psnr`
  - PSNR mean 16.654, wall 810.21s, `used_anneal_steps=200`
- 해석: 품질 비열등 실패(−1.04 dB). 조기종료 전이므로 시간 이득도 없음.
- 조치: K2 보수화(tc=0.92, f=3/2, ema=0.2/0.3) 또는 조기종료 적용으로 실제 스텝 절감 확인.

### 16) 실수/오류 정리(override 파손 및 잘못된 실행 템플릿)
- 증상: 로그가 `name=demo`, `task_group=pixel`, `k2_scheduler_config: null`, `data.end_id=100` 등으로 표시.
  - 원인: 멀티라인 커맨드 복붙 파손(역슬래시/개행)으로 Hydra override 미적용.
  - 해결: 반드시 단일행 명령으로 실행. `task_group=pixel_hmc`를 명시하고 `name=<런명>` 확인.
- 확인 스니펫
  ```bash
  ps -eo pid,etimes,cmd | grep -E 'posterior_sample.py|time_ab' | grep -v grep
  tail -n 60 /home/elicer/our_win/*.txt
  grep -n 'task_group:' results/<run>/config.yaml
  ```

### 17) σ(시그마) 로그 분석 포인트
- 스윗 스팟: 중간 σ 구간에서 PSNR이 피크(예: ~19 dB) 후 tail(σ≈0.05)에서 소폭 하락 가능 → 정상.
- 명령(최근 σ/스텝 확인)
  ```bash
  RUN=/home/elicer/DAPS_local/results/time_ab_k2_psnr_early
  tail -n 30 "$RUN/events.jsonl" | sed -n 's/.*"sigma_used": \([^}]*\).*/sigma_used=\1/p; s/.*"next_sigma": \([^,}]*\).*/next_sigma=\1/p'
  awk '/"mode": "used"/{c++; printf("%03d %s\n",c,$0)}' "$RUN/events.jsonl" | sed -n '90,110p'
  ```

### 18) 최근 실행 결과(0–20 슬라이스, Euler, batch=10, runs=1) 요약
- 기준 HMC (PSNR만 평가)
  - 런: `results/time_ab_hmc_psnr`
  - 설정: `task_group=pixel_hmc`, ODE=5, ANN=200, no‑op K2(cess=1.0,f=1.0), `+eval_fn_list='[psnr]'`
  - 결과: PSNR mean 17.696, wall 811.08s, `used_anneal_steps=200`
- K2‑CESS fast (PSNR만 평가)
  - 런: `results/time_ab_k2_psnr` (tc=0.90, f=3, ema=0.2, minσ=0.05)
  - 결과: PSNR mean 16.654, wall 810.21s, `used_anneal_steps=200`
- K2‑CESS quality (PSNR+SSIM+LPIPS)
  - 런: `results/ab_k2_cess_euler_min005_tc095_f2_s0_20` (tc=0.95, f=2, ema=0.2, minσ=0.05)
  - 결과: PSNR mean 16.119, wall 873.36s (비열등 실패)
- K2 스모크(참고; 평가 PSNR만, minσ=0.10)
  - 런: `results/pr_k2_cess_acc_smoke_min01`
  - 결과: PSNR mean 17.636, wall 802.77s (평가 항목 불일치로 공정비교 아님)
- 주의: `demo`로 뜬 로그는 `task_group=pixel`, `k2_scheduler_config: null`로 잘못 실행된 사례(override 파손). 공정 비교에서 제외.

