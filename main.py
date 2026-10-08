import os
import argparse
import json
import torch.multiprocessing as mp
from src.utils import load_config
from src.train_simclr import train_simclr
from src.train_supervised import train_supervised

mp.set_sharing_strategy('file_system')

def run_all_experiments(force=False):
    config = load_config('config.json')
    noise_levels = config['noise_levels']

    # Three-stage design:
    #   baseline        - ImageNet-pretrained ResNet-18, supervised only
    #   simclr_finetune - SimCLR-pretrained ResNet-18, plain supervised fine-tuning
    #   full_proposed   - SimCLR + class-wise GMM + class-quantile curriculum
    #                     + clean-effective class balancing + Focal/SCE loss
    stages = [
        'baseline',
        'simclr_finetune',
        'full_proposed'
    ]

    # 1. Train SimCLR once
    if force or not os.path.exists('checkpoints/simclr_best.pth'):
        print("=== Starting SimCLR Pretraining ===")
        train_simclr(config)
    else:
        print("=== SimCLR Checkpoint Found, Skipping Pretraining ===")

    # 2. Run all stages across all noise levels

    # Load existing results to allow resuming
    completed_runs = set()
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            results = json.load(f)
            completed_runs = {r['run_name'] for r in results}

    for stage in stages:
        for noise in noise_levels:
            run_name = f"{stage}_noise_{int(noise*100)}"
            if run_name in completed_runs and not force:
                print(f"=== Skipping {run_name} (already completed) ===")
                continue
            train_supervised(config, noise, stage, run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all', action='store_true', help="Run the full pipeline")
    parser.add_argument('--simclr_only', action='store_true', help="Run only SimCLR pretraining")
    parser.add_argument('--stage', type=str, default=None, help="Run specific stage")
    parser.add_argument('--noise', type=float, default=0.0, help="Noise level")
    parser.add_argument('--force', action='store_true', help="Re-run everything, ignoring existing checkpoints and results")

    args = parser.parse_args()

    if args.run_all:
        run_all_experiments(force=args.force)
    elif args.simclr_only:
        config = load_config('config.json')
        train_simclr(config)
    elif args.stage:
        config = load_config('config.json')
        run_name = f"{args.stage}_noise_{int(args.noise*100)}"
        train_supervised(config, args.noise, args.stage, run_name)
    else:
        print("Please provide an argument. Use --help for options.")