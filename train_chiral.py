import argparse
from rdkit import Chem
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import copy
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
import multiprocessing
import json
from rdkit.Chem import rdMolAlign as MA
from builtins import enumerate
import os
import argparse
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from natgen.e2c.dataset import PygGeomDataset
from natgen.model.gnn_chiral import ChiralGNN
from torch.optim.lr_scheduler import LambdaLR
from natgen.utils.utils import (
    Cosinebeta,
    WarmCosine,
    set_rdmol_positions,
    get_best_rmsd,
    evaluate_distance,
)
from collections import defaultdict
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, random_split
from torch.cuda.amp import GradScaler, autocast  
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdchem
import itertools
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import RemoveHs
import copy
from rdkit import Chem
import torch.nn.functional as F
import io
import json
import multiprocessing
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from torch_geometric.utils import dropout_adj
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import time
import signal
from torch.utils.data import ConcatDataset
def fix_key_name(state_dict, add_module=True):
    new_state_dict = {}
    for key, value in state_dict.items():
        if add_module and not key.startswith('module.'):
            new_key = 'module.' + key
        elif not add_module and key.startswith('module.'):
            new_key = key[7:]  
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
def handle_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    if isinstance(obj, np.number):
        return obj.item()  
    raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))
def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {} local rank {}): {}".format(
            args.rank, args.local_rank, "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
def calculate_edges_per_graph_undirected_modified(edge_index, batch):
    num_graphs = batch.num_graphs
    edges_per_graph = [0] * num_graphs
    if not hasattr(batch, 'batch'):
        raise AttributeError("batch 对象中缺少 'batch' 属性。")
    for i in range(edge_index.shape[1]):
        src, dest = edge_index[:, i]
        graph_id = batch.batch[src].item() 
        edges_per_graph[graph_id] += 1
    return torch.tensor(edges_per_graph)
def freeze_parameters(model, freeze_modules=None):
    for name, param in model.named_parameters():
        if freeze_modules and not any(name.startswith(module) for module in freeze_modules):
            param.requires_grad = False
def one_hot_labels(labels, num_classes=3):
    return F.one_hot(labels, num_classes=num_classes).float()
def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            use_mixup = args.inter_mixup_weight > 0 or args.intra_mixup_weight > 0
            if use_mixup:
                model_outputs = model(batch)
                if len(model_outputs) == 2:
                    chirality_list, extra_output = model_outputs
                    inter_mixup_logits, intra_mixup_logits = None, None
                else:
                    chirality_list, extra_output, inter_mixup_logits, intra_mixup_logits = model_outputs
                if not isinstance(chirality_list, list):
                    chirality_list = [chirality_list]
                optimizer.zero_grad()
                if args.distributed:
                    total_loss, loss_dict = model.module.compute_loss(
                        chirality_list, extra_output, batch, args, 
                        inter_mixup_logits, intra_mixup_logits
                    )
                else:
                    total_loss, loss_dict = model.compute_loss(
                        chirality_list, extra_output, batch, args,
                        inter_mixup_logits, intra_mixup_logits
                    )
                total_loss.backward()
                if args.grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                scheduler.step()
                for k, v in loss_dict.items():
                    loss_accum_dict[k] += v.detach().item()
                if step % args.log_interval == 0:
                    description = f"Loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                    if 'loss_inter_mixup' in loss_dict and args.inter_mixup_weight > 0:
                        description += f" Inter({args.inter_mixup_weight:.3f}): {loss_accum_dict['loss_inter_mixup'] / (step + 1):6.4f}"
                    if 'loss_intra_mixup' in loss_dict and args.intra_mixup_weight > 0:
                        description += f" Intra({args.intra_mixup_weight:.3f}): {loss_accum_dict['loss_intra_mixup'] / (step + 1):6.4f}"
                    description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
                    pbar.set_description(description)
            else:
                model_outputs = model(batch, sample=False)
                if len(model_outputs) == 2:
                    chirality_list, extra_output = model_outputs
                else:
                    chirality_list, extra_output = model_outputs[0], model_outputs[1]
                optimizer.zero_grad()
                if args.distributed:
                    loss, loss_dict = model.module.compute_loss(chirality_list, extra_output, batch, args)
                else:
                    loss, loss_dict = model.compute_loss(chirality_list, extra_output, batch, args)
                total_loss = loss
                total_loss.backward()
                if args.grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                scheduler.step()
                for k, v in loss_dict.items():
                    loss_accum_dict[k] += v.detach().item()
                if step % args.log_interval == 0:
                    description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                    description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
                    description += f" vae_beta: {args.vae_beta:6.4f}"
                    pbar.set_description(description)
    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict
def evaluate(model, device, loader, args):
    model.eval()
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%',
                 '40-50%', '50-60%', '60-70%', '70-80%']
    bin_stats = {label: [0, 0] for label in bin_labels}
    bin_stats['80-100%'] = [0, 0]
    total_correct_molecules = 0  # For chirality_accuracy_100_percent
    total_correct_nodes = 0      # For node_accuracy
    total_molecules = 0          # Molecules with chiral atoms
    total_nodes = 0              # Total nodes
    loss_accum_dict = defaultdict(float)
    total_steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            model_outputs = model(batch, sample=True)
            if len(model_outputs) == 2:
                chirality_list, extra_output = model_outputs
            else:
                chirality_list, extra_output = model_outputs[0], model_outputs[1]
            if args.distributed:
                loss, loss_dict = model.module.compute_loss(chirality_list, extra_output, batch, args)
            else:
                loss, loss_dict = model.compute_loss(chirality_list, extra_output, batch, args)
            preds = torch.argmax(chirality_list[-1], dim=-1)
            labels = batch.chiral
            molecule_indices = batch.batch
            total_correct_nodes += (preds == labels).sum().item()
            total_nodes += labels.size(0)
            for mol_id in molecule_indices.unique():
                node_mask = (molecule_indices == mol_id)
                mol_preds = preds[node_mask]
                mol_labels = labels[node_mask]
                chiral_mask = (mol_labels == 1) | (mol_labels == 2)
                if not chiral_mask.any():
                    continue  
                n_chiral = chiral_mask.sum().item()
                ratio = n_chiral / len(mol_labels)
                if ratio >= 0.8:
                    bin_name = '80-100%'
                else:
                    for i, edge in enumerate(bin_edges[1:]):
                        if ratio < edge:
                            bin_name = bin_labels[i]
                            break
                is_correct = (mol_preds[chiral_mask] == mol_labels[chiral_mask]).all()
                if is_correct:
                    bin_stats[bin_name][0] += 1
                    total_correct_molecules += 1
                bin_stats[bin_name][1] += 1
                total_molecules += 1
            for k, v in loss_dict.items():
                loss_accum_dict[k] += v.item()
            total_steps += 1
    accuracy_report = []
    for bin_name in (bin_labels + ['80-100%']):
        correct, total = bin_stats[bin_name]
        accuracy = correct / total if total > 0 else 0.0
        accuracy_report.append({
            'ratio_bin': bin_name,
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        })
    print("\nAccuracy by Chiral Atom Ratio:")
    print("| Ratio Range | Correct    | Total     | Accuracy |")
    print("|-------------|------------|-----------|----------|")
    for item in accuracy_report:
        accuracy_str = f"{item['accuracy']:.2%}" if item['total'] > 0 else "N/A"
        print(f"| {item['ratio_bin']:11} | {item['correct']:8} | {item['total']:8} | {accuracy_str:8} |")
    chirality_accuracy_100_percent = total_correct_molecules / total_molecules if total_molecules > 0 else 0.0
    node_accuracy = total_correct_nodes / total_nodes if total_nodes > 0 else 0.0
    for k in loss_accum_dict:
        loss_accum_dict[k] /= total_steps
    return chirality_accuracy_100_percent, node_accuracy, loss_accum_dict
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.15)
    parser.add_argument("--dropnode-rate", type=float, default=0.15)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--mlp-hidden-size", type=int, default=256)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint/NatGen/")
    parser.add_argument("--log_dir", type=str, default="./checkpoint/NatGen/NatGen-LOG/")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--encoder-dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=True)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--base-path", type=str, default="./dataset/")
    parser.add_argument("--dataset-name", type=str, choices=["Crystal", "Coconut","MIX_Chiral","CSD_Chiral","COD_Chiral","PDB_Chiral"], default="Coconut")
    parser.add_argument("--data-split", type=str, choices=["Crystal", "Coconut","MIX_Chiral","CSD_Chiral","COD_Chiral","PDB_Chiral"], default="Coconut")
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--enable-tb", action="store_true", default=True)
    parser.add_argument("--aux-loss", type=float, default=0.2)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--extend-edge", action="store_true", default=True)
    parser.add_argument("--reuse-prior", action="store_true", default=True)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--vae-beta", type=float, default=1)
    parser.add_argument("--sample-beta", type=float, default=1.0)
    parser.add_argument("--vae-beta-max", type=float, default=0.03)#0.03
    parser.add_argument("--vae-beta-min", type=float, default=0.0001)#0.0001
    parser.add_argument("--pred-pos-residual", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--shared-decoder", action="store_true", default=False)
    parser.add_argument("--shared-output", action="store_true", default=True)
    parser.add_argument("--clamp-dist", type=float, default=None)
    parser.add_argument("--use-global", action="store_true", default=False)
    parser.add_argument("--sg-pos", action="store_true", default=False)
    parser.add_argument("--remove-hs", action="store_true", default=True)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--use-ss", action="store_true", default=False)
    parser.add_argument("--rand-aug", action="store_true", default=False)
    parser.add_argument("--not-origin", action="store_true", default=False)
    parser.add_argument("--ang-lam", type=float, default=0.2)
    parser.add_argument("--bond-lam", type=float, default=0.1)
    parser.add_argument("--no-3drot", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--use-ff", action="store_true", default=False)
    parser.add_argument("--chiral-loss-weight", type=float, default=1)
    parser.add_argument('--mixup-ratio', type=float, default=0.5)
    parser.add_argument("--main-loss-weight-stage1", type=float, default=1.0, help="Weight for main loss in stage 1")
    parser.add_argument("--inter-mixup-weight-stage1", type=float, default=0.0, help="Weight for inter mixup loss in stage 1")
    parser.add_argument("--intra-mixup-weight-stage1", type=float, default=0.0, help="Weight for intra mixup loss in stage 1")
    parser.add_argument("--main-loss-weight-stage2", type=float, default=0.5, help="Weight for main loss in stage 2")
    parser.add_argument("--inter-mixup-weight-stage2", type=float, default=0.25, help="Weight for inter mixup loss in stage 2")
    parser.add_argument("--intra-mixup-weight-stage2", type=float, default=0.25, help="Weight for intra mixup loss in stage 2")
    parser.add_argument("--stage-switch-epoch", type=int, default=301, help="Epoch to switch from stage 1 to stage 2")
    parser.add_argument("--main-loss-weight", type=float, default=0.5, help="Weight for main loss (original graph) - overrides stage settings")
    parser.add_argument("--inter-mixup-weight", type=float, default=0.25, help="Weight for inter mixup loss - overrides stage settings")
    parser.add_argument("--intra-mixup-weight", type=float, default=0.25, help="Weight for intra mixup loss - overrides stage settings")
    args = parser.parse_args()
    if args.main_loss_weight is not None or args.inter_mixup_weight is not None or args.intra_mixup_weight is not None:
        args.use_stage_weights = False
        if args.main_loss_weight is None:
            args.main_loss_weight = 1.0
        if args.inter_mixup_weight is None:
            args.inter_mixup_weight = 0.0
        if args.intra_mixup_weight is None:
            args.intra_mixup_weight = 0.0
        total_weight = args.main_loss_weight + args.inter_mixup_weight + args.intra_mixup_weight
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        args.main_loss_weight /= total_weight
        args.inter_mixup_weight /= total_weight
        args.intra_mixup_weight /= total_weight
        print(f"Using fixed weights - Main: {args.main_loss_weight:.3f}, Inter: {args.inter_mixup_weight:.3f}, Intra: {args.intra_mixup_weight:.3f}")
    else:
        args.use_stage_weights = True
        total_weight_s1 = args.main_loss_weight_stage1 + args.inter_mixup_weight_stage1 + args.intra_mixup_weight_stage1
        if total_weight_s1 <= 0:
            raise ValueError("Stage 1 total weight must be positive")
        args.main_loss_weight_stage1 /= total_weight_s1
        args.inter_mixup_weight_stage1 /= total_weight_s1
        args.intra_mixup_weight_stage1 /= total_weight_s1
        total_weight_s2 = args.main_loss_weight_stage2 + args.inter_mixup_weight_stage2 + args.intra_mixup_weight_stage2
        if total_weight_s2 <= 0:
            raise ValueError("Stage 2 total weight must be positive")
        args.main_loss_weight_stage2 /= total_weight_s2
        args.inter_mixup_weight_stage2 /= total_weight_s2
        args.intra_mixup_weight_stage2 /= total_weight_s2
        print(f"Using stage weights:")
        print(f"  Stage 1 (epochs 1-{args.stage_switch_epoch-1}): Main={args.main_loss_weight_stage1:.3f}, Inter={args.inter_mixup_weight_stage1:.3f}, Intra={args.intra_mixup_weight_stage1:.3f}")
        print(f"  Stage 2 (epochs {args.stage_switch_epoch}+): Main={args.main_loss_weight_stage2:.3f}, Inter={args.inter_mixup_weight_stage2:.3f}, Intra={args.intra_mixup_weight_stage2:.3f}")
        args.main_loss_weight = args.main_loss_weight_stage1
        args.inter_mixup_weight = args.inter_mixup_weight_stage1
        args.intra_mixup_weight = args.intra_mixup_weight_stage1
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print(f"Created directory {args.checkpoint_dir}")
    else:
        print(f"Directory {args.checkpoint_dir} already exists")
    init_distributed_mode(args)
    print(args)
    CosineBeta = Cosinebeta(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    dataset = PygGeomDataset(
        root="dataset",
        dataset=args.dataset_name,
        base_path=args.base_path,
        seed=args.seed,
        extend_edge=args.extend_edge,
        data_split=args.data_split,
        remove_hs=args.remove_hs,
    )
    split_idx = dataset.get_idx_split()
    print(f"🚀 Available data - Train: {len(split_idx['train'])}, Valid: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
    dataset_train = (
        dataset[split_idx["train"]]
        if not args.train_subset
        else dataset[split_idx["train"]]
    )
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    train_loader = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
    )
    train_loader_dev = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Training set size: {len(train_loader.dataset)} graphs")
    print(f"Validation set size: {len(valid_loader.dataset)} graphs")
    print(f"Testing set size: {len(test_loader.dataset)} graphs")
    shared_params = {
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_layers": args.mlp_layers,
        "latent_size": args.latent_size,
        "use_layer_norm": args.use_layer_norm,
        "num_message_passing_steps": args.num_layers,
        "global_reducer": args.global_reducer,
        "node_reducer": args.node_reducer,
        "dropedge_rate": args.dropedge_rate,
        "dropnode_rate": args.dropnode_rate,
        "dropout": args.dropout,
        "layernorm_before": args.layernorm_before,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "vae_beta": args.vae_beta,
        "decoder_layers": args.decoder_layers,
        "reuse_prior": args.reuse_prior,
        "cycle": args.cycle,
        "pred_pos_residual": args.pred_pos_residual,
        "node_attn": args.node_attn,
        "global_attn": args.global_attn,
        "shared_decoder": args.shared_decoder,
        "use_global": args.use_global,
        "sg_pos": args.sg_pos,
        "shared_output": args.shared_output,
        "use_ss": args.use_ss,
        "rand_aug": args.rand_aug,
        "no_3drot": args.no_3drot,
        "not_origin": args.not_origin,
    }
    model = ChiralGNN(
        batch_size=args.batch_size, 
        mixup_ratio=args.mixup_ratio, 
        inter_mixup_weight=args.inter_mixup_weight,
        intra_mixup_weight=args.intra_mixup_weight,
        **shared_params
    ).to(device)
    model_without_ddp = model
    args.disable_tqdm = False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0
    if args.eval_from is not None:
        assert os.path.exists(args.eval_from), "No Pretrained Model"
        checkpoint = torch.load(args.eval_from, map_location="cpu")["model_state_dict"]
        model_state = model_without_ddp.state_dict()
        new_model_state = {}
        for name, param in model_state.items():
            if name in checkpoint and checkpoint[name].shape == param.shape:
                new_model_state[name] = checkpoint[name]
            else:
                new_model_state[name] = param
        model_without_ddp.load_state_dict(new_model_state, strict=False)
        print("Finetuning with partial pretrained weights. GOOD LUCK!!!!!")
    restore_fn = os.path.join(args.checkpoint_dir, "checkpoint_x.pt")
    if args.restore:
        if os.path.exists(restore_fn):
            print(f"Restore from {restore_fn}")
            restore_checkpoint = torch.load(restore_fn, map_location=torch.device('cpu'))
            model_state_dict = restore_checkpoint["model_state_dict"]
            if args.distributed:
                model_state_dict = fix_key_name(model_state_dict, add_module=True)
            else:
                model_state_dict = fix_key_name(model_state_dict, add_module=False)
            model_without_ddp.load_state_dict(model_state_dict, strict=False)
            model_without_ddp.to(device)
            print("Model restored successfully.")
        else:
            print("No checkpoint found at", restore_fn)
            args.restore = False
    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")
    if args.use_adamw:
        optimizer = optim.AdamW(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
        )
    if not args.lr_warmup:
        scheduler = LambdaLR(optimizer, lambda x: 1.0)
    else:
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
        scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
    if args.restore:
        optimizer.load_state_dict(restore_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(restore_checkpoint["scheduler_state_dict"])
    if args.log_dir and args.enable_tb:
        tb_writer = SummaryWriter(args.log_dir)
    start_epoch = restore_checkpoint["epoch"] if args.restore else 0
    if args.log_dir:
        hyperparams = vars(args)
        with io.open(os.path.join(args.log_dir, "log.txt"), "w", encoding="utf8", newline="\n") as tgt:
            print(json.dumps({"Hyperparameters": hyperparams}, indent=4), file=tgt)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if args.use_stage_weights and epoch == args.stage_switch_epoch:
            print(f"\n🔄 Switching to Stage 2 weights at epoch {epoch}")
            args.main_loss_weight = args.main_loss_weight_stage2
            args.inter_mixup_weight = args.inter_mixup_weight_stage2
            args.intra_mixup_weight = args.intra_mixup_weight_stage2
            model_without_ddp.inter_mixup_weight = args.inter_mixup_weight
            model_without_ddp.intra_mixup_weight = args.intra_mixup_weight
            print(f"Updated weights - Main: {args.main_loss_weight:.3f}, Inter: {args.inter_mixup_weight:.3f}, Intra: {args.intra_mixup_weight:.3f}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        CosineBeta.step(epoch - 1)
        if args.use_stage_weights:
            stage = 1 if epoch < args.stage_switch_epoch else 2
            print("=====Epoch {} (Stage {})".format(epoch, stage))
        else:
            print("=====Epoch {}".format(epoch))
        print("Training...")
        loss_dict = train(model, device, train_loader, optimizer, scheduler, args)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_without_ddp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": args,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))
        if args.enable_tb and tb_writer is not None:
            for k, v in loss_dict.items():
                tb_writer.add_scalar(f"training/{k}", v, epoch)
            tb_writer.add_scalar("training/loss_prior_chirality", loss_dict.get("loss_prior_chirality", 0.0), epoch)
            tb_writer.add_scalar("training/chiral_accuracy_loss", loss_dict.get("loss_chirality_last", 0.0), epoch)
            tb_writer.add_scalar("training/loss_kld", loss_dict.get("loss_kld", 0.0), epoch)
            tb_writer.add_scalar("training/loss_task1", loss_dict.get("loss_task1", 0.0), epoch)
            tb_writer.add_scalar("training/loss_task2", loss_dict.get("loss_task2", 0.0), epoch)
            tb_writer.add_scalar("training/total_loss", loss_dict.get("loss", 0.0), epoch)
            if "loss_inter_mixup" in loss_dict:
                tb_writer.add_scalar("training/loss_inter_mixup", loss_dict["loss_inter_mixup"], epoch)
            if "loss_intra_mixup" in loss_dict:
                tb_writer.add_scalar("training/loss_intra_mixup", loss_dict["loss_intra_mixup"], epoch)
            tb_writer.add_scalar("weights/main_loss_weight", args.main_loss_weight, epoch)
            tb_writer.add_scalar("weights/inter_mixup_weight", args.inter_mixup_weight, epoch)
            tb_writer.add_scalar("weights/intra_mixup_weight", args.intra_mixup_weight, epoch)
            if args.use_stage_weights:
                current_stage = 1 if epoch < args.stage_switch_epoch else 2
                tb_writer.add_scalar("stage/current_stage", current_stage, epoch)
        print("Evaluating")
        chirality_accuracy_chiral, node_accuracy, eval_test_loss_dict = evaluate(model, device, valid_loader, args)
        print(f"Chirality Accuracy 100%: {chirality_accuracy_chiral:.4f}")
        print(f"Node Accuracy: {node_accuracy:.4f}")
        for k, v in eval_test_loss_dict.items():
            print(f"{k}: {v:.4f}")
        if args.log_dir:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logs = {
                "Time": current_time,
                "Epoch": epoch + 1,
                "Chirality_Accuracy_100%": chirality_accuracy_chiral,
                "Node_Accuracy": node_accuracy,
            }
            for k, v in eval_test_loss_dict.items():
                logs[k] = v
            with io.open(
                os.path.join(args.log_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs, default=handle_numpy), file=tgt)
        if args.enable_tb and tb_writer is not None:
            tb_writer.add_scalar("Chirality Accuracy Chiral/Test", chirality_accuracy_chiral, epoch)
            tb_writer.add_scalar("Node Accuracy/Test", node_accuracy, epoch)
            for k, v in eval_test_loss_dict.items():
                tb_writer.add_scalar(f"evaluating/{k}", v, epoch)
    if args.log_dir and args.enable_tb:
        tb_writer.close()
    if args.distributed:
        torch.distributed.destroy_process_group()
    print("Finished training!")
if __name__ == "__main__":
    main()
