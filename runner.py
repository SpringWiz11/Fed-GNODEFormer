#!/usr/bin/env python
import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run federated or centralized GNN training")
    
    parser.add_argument("--mode", type=str, choices=["federated", "centralized"], required=True,
                        help="Training mode: federated or centralized")
    parser.add_argument("--dataset", type=str, default="cora", 
                        help="Dataset name")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use GPU if available")
    
    fed_group = parser.add_argument_group("Federated training options")
    fed_group.add_argument("--global_rounds", type=int, default=100,
                        help="Number of global communication rounds")
    fed_group.add_argument("--local_step", type=int, default=3,
                        help="Number of local steps per round")
    fed_group.add_argument("--n_trainer", type=int, default=5,
                        help="Number of trainers/clients")
    fed_group.add_argument("--alpha", type=float, default=0.01,
                        help="Dirichlet distribution parameter alpha")
    fed_group.add_argument("--repeat_time", type=int, default=10,
                        help="Number of repeats for federated training")
    fed_group.add_argument("--lr_fed", type=float, default=0.05,
                        help="Learning rate")
    fed_group.add_argument("--weight_decay_fed", type=float, default=5e-3,
                        help="Weight decay")
    
    cent_group = parser.add_argument_group("Centralized training options") 
    cent_group.add_argument("--image", type=int, default=0,
                        help="Image index for signal datasets")
    cent_group.add_argument("--epoch", type=int, default=1000,
                        help="Maximum number of training epochs")
    cent_group.add_argument("--nclass", type=int, default=7,
                        help="Number of classes in the dataset")
    cent_group.add_argument("--lr_cent", type=float, default=0.0002,
                        help="Learning rate")
    cent_group.add_argument("--weight_decay_cent", type=float, default=0.0001,
                        help="Weight decay")
    
    model_group = parser.add_argument_group("Model hyperparameters")
    model_group.add_argument("--nlayer", type=int, default=2,
                        help="Number of layers")
    model_group.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension size")
    model_group.add_argument("--num_heads", type=int, default=2,
                        help="Number of attention heads")
    model_group.add_argument("--rk", type=int, default=2,
                        help="Rank parameter (federated only)")
    model_group.add_argument("--tran_dropout", type=float, default=0.2,
                        help="Transformer dropout rate")
    model_group.add_argument("--feat_dropout", type=float, default=0.6,
                        help="Feature dropout rate")
    model_group.add_argument("--prop_dropout", type=float, default=0.2,
                        help="Propagation dropout rate")
    model_group.add_argument("--norm", type=str, default="none",
                        help="Normalization type")
    
    return parser.parse_args()


def run_federated(args):
    """Run federated training with the provided arguments"""
    cmd = [
        "python", "src/Fed_GNODEFormer/fed_main.py",
        "-d", args.dataset,
        "-c", str(args.global_rounds),
        "-i", str(args.local_step),
        "-n", str(args.n_trainer),
        "-a", str(args.alpha),
        "-nl", str(args.nlayer),
        "-hd", str(args.hidden_dim),
        "-nh", str(args.num_heads),
        "-rk", str(args.rk),
        "-td", str(args.tran_dropout),
        "-fd", str(args.feat_dropout),
        "-pd", str(args.prop_dropout),
        "-no", args.norm,
        "-lr", str(args.lr_fed),
        "-wd", str(args.weight_decay_fed),
        "-r", str(args.repeat_time)
    ]
    
    if args.gpu:
        cmd.append("-g")
    
    print(f"Running federated training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_centralized(args):
    """Run centralized training with the provided arguments"""
    cmd = [
        "python", "src/GNODEFormer/centralised_main.py",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--epoch", str(args.epoch),
        "--nclass", str(args.nclass),
        "--nlayer", str(args.nlayer),
        "--hidden_dim", str(args.hidden_dim),
        "--num_heads", str(args.num_heads),
        "--tran_dropout", str(args.tran_dropout),
        "--feat_dropout", str(args.feat_dropout),
        "--prop_dropout", str(args.prop_dropout),
        "--norm", args.norm,
        "--lr", str(args.lr_cent),
        "--weight_decay", str(args.weight_decay_cent),
        "--rk", str(args.rk)
    ]
    
    if args.gpu:
        cmd.append("--gpu")
    
    print(f"Running centralized training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    args = parse_args()
    print(f"Starting {args.mode} training for dataset: {args.dataset}")
    
    if args.mode == "federated":
        run_federated(args)
    else:
        run_centralized(args)
    
    print(f"{args.mode.capitalize()} training completed!")


if __name__ == "__main__":
    main()