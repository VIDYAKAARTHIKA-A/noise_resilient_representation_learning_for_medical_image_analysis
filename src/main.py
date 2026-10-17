import os
import json
import argparse
import torch.multiprocessing as mp
from src.utils import load_config
from src.train_simclr import train_simclr
from src.train_supervised import train_supervised, STAGES

mp.set_sharing_strategy('file_system')


def make_run_name(stage, noise, noise_type, val_mode, seed):
    nt = 'none' if noise == 0 else noise_type          # at 0% noise the noise type is irrelevant
    return f"{stage}_{nt}_n{int(round(noise * 100))}_{val_mode}_s{seed}"


def needs_simclr(config, stages):
    return (any(STAGES[s]['init'] == 'simclr' for s in stages)
            or config.get('v2', {}).get('ref_encoder') == 'simclr')


def run_all(config, args, force=False):
    stages = args.stages or config['stages']
    seeds = args.seeds or config['seeds']
    noise_levels = args.noise_levels or config['noise_levels']
    noise_types = args.noise_types or config['noise_types']
    val_modes = args.val_modes or config['val_modes']

    if needs_simclr(config, stages) and (force or not os.path.exists('checkpoints/simclr_best.pth')):
        print("=== SimCLR pretraining ===")
        train_simclr(config)

    done = set()
    if os.path.exists('results.json') and not force:
        with open('results.json') as f:
            done = {r['run_name'] for r in json.load(f)}

    seen = set()
    for seed in seeds:
        for val_mode in val_modes:
            for noise_type in noise_types:
                for noise in noise_levels:
                    for stage in stages:
                        name = make_run_name(stage, noise, noise_type, val_mode, seed)
                        if name in seen:               # 0% noise would otherwise repeat per noise type
                            continue
                        seen.add(name)
                        if name in done:
                            print(f"=== Skipping {name} (done) ===")
                            continue
                        train_supervised(config, noise, stage, name, noise_type, val_mode, seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_all', action='store_true')
    p.add_argument('--simclr_only', action='store_true')
    p.add_argument('--stage', type=str, default=None, help=f"one of {list(STAGES)}")
    p.add_argument('--noise', type=float, default=0.0)
    p.add_argument('--noise_type', type=str, default='uniform_global')
    p.add_argument('--val_mode', type=str, default='clean', choices=['clean', 'noisy', 'last'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true')
    # overrides for --run_all
    p.add_argument('--stages', nargs='+', default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--noise_levels', nargs='+', type=float, default=None)
    p.add_argument('--noise_types', nargs='+', default=None)
    p.add_argument('--val_modes', nargs='+', default=None)
    args = p.parse_args()

    config = load_config('config.json')
    if args.run_all:
        run_all(config, args, force=args.force)
    elif args.simclr_only:
        train_simclr(config)
    elif args.stage:
        name = make_run_name(args.stage, args.noise, args.noise_type, args.val_mode, args.seed)
        train_supervised(config, args.noise, args.stage, name, args.noise_type, args.val_mode, args.seed)
    else:
        p.print_help()
