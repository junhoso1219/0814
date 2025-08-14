#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import csv
import yaml


def find_runs(results_dir: Path):
    for root, dirs, files in os.walk(results_dir):
        if 'metrics.json' in files and 'config.yaml' in files:
            yield Path(root)


def read_metrics(run_dir: Path):
    metrics = json.load(open(run_dir / 'metrics.json'))
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    def get_main(values):
        # prefer mean if available else average of cmp list
        if 'mean' in values:
            return sum(values['mean']) / len(values['mean'])
        cmp_key = None
        # lpips is min, psnr/ssim are max in repo
        for key in ('psnr', 'ssim', 'lpips'):
            if key in metrics:
                pass
        return None
    row = {
        'name': cfg.get('name'),
        'task_group': cfg.get('task_group'),
        'data': cfg.get('data', {}).get('name'),
        'model': cfg.get('model', {}).get('name'),
        'ode_steps': cfg.get('sampler', {}).get('diffusion_scheduler_config', {}).get('num_steps'),
        'anneal_steps': cfg.get('sampler', {}).get('annealing_scheduler_config', {}).get('num_steps'),
    }
    # pull means/stds
    for k in ('psnr', 'ssim', 'lpips'):
        if k in metrics:
            row[f'{k}_mean'] = sum(metrics[k]['mean']) / len(metrics[k]['mean'])
            row[f'{k}_std'] = sum(metrics[k]['std']) / len(metrics[k]['std'])
        else:
            row[f'{k}_mean'] = ''
            row[f'{k}_std'] = ''
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, required=True)
    ap.add_argument('--output_csv', type=str, required=True)
    args = ap.parse_args()

    runs = list(find_runs(Path(args.results_dir)))
    rows = []
    for r in runs:
        try:
            rows.append(read_metrics(r))
        except Exception as e:
            print(f'skip {r}: {e}')

    fieldnames = ['name', 'task_group', 'data', 'model', 'ode_steps', 'anneal_steps',
                  'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std', 'lpips_mean', 'lpips_std']
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'Wrote {args.output_csv} with {len(rows)} rows')


if __name__ == '__main__':
    main()


