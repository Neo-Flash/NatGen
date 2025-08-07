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
from natgen.model.gnn_conf import ConfGNN, one_hot_atoms, one_hot_bonds
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
def fix_key_name(state_dict, add_module=True):
    """调整权重键名，以匹配模型是否使用了DDP"""
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
def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum_dict = defaultdict(float)
    line_loss_accum_dict = defaultdict(float)
    mixup_loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        lambda_ = args.mixup_ratio
        index = torch.randperm(batch.x.shape[0])
        index = torch.cat((index, torch.arange(batch.x.shape[0], batch.x.shape[0])), dim=0)
        index_new = torch.zeros(index.shape[0], dtype=torch.long)
        index_new[index] = torch.arange(0, index.shape[0])
        pos = batch.pos.clone().detach()
        mixup_pos = lambda_ * pos + (1-lambda_) * pos[index]
        batch.mixup_pos = mixup_pos
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            atom_pred_list, extra_output, line_graph_pred_list, mixup_pred_list = model(batch)
            optimizer.zero_grad()
            if args.distributed:
                loss, loss_dict = model.module.compute_loss(atom_pred_list, extra_output, batch, args)
                line_graph_loss, line_graph_loss_dict = model.module.compute_line_loss(line_graph_pred_list, extra_output, batch, args)
                mixup_loss, mixup_loss_dict = model.module.compute_mixup_loss(mixup_pred_list, extra_output, batch, args)
            else:
                loss, loss_dict = model.compute_loss(atom_pred_list, extra_output, batch, args)
                line_graph_loss, line_graph_loss_dict = model.compute_line_loss(line_graph_pred_list,extra_output, batch, args)
                mixup_loss, mixup_loss_dict = model.compute_mixup_loss(mixup_pred_list, extra_output, batch, args)
            total_loss = args.main_loss * loss  + args.mixup_loss * mixup_loss + args.line_loss * line_graph_loss
            total_loss.backward()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            scheduler.step()
            for k, v in loss_dict.items():
                loss_accum_dict[k] += v.detach().item()
            for k, v in mixup_loss_dict.items():
                mixup_loss_accum_dict[k] += v.detach().item()
            for k, v in line_graph_loss_dict.items():
                line_loss_accum_dict[k] += v.detach().item()
            if step % args.log_interval == 0:
                description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
                description += f" vae_beta: {args.vae_beta:6.4f}"
                pbar.set_description(description)
    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    for k in mixup_loss_accum_dict.keys():
        mixup_loss_accum_dict[k] /= step + 1
    for k in line_loss_accum_dict.keys():
        line_loss_accum_dict[k] /= step + 1
    return loss_accum_dict, mixup_loss_accum_dict,  line_loss_accum_dict
def get_rmsd_min(inputargs):
    mols, use_ff, threshold = inputargs
    gen_mols, ref_mols = mols
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        if use_ff:
            MMFFOptimizeMolecule(gen_mol_c)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()
def evaluate(model, device, loader, args):
    model.eval()
    mol_labels = []
    mol_preds = []
    loss_accum_dict = defaultdict(float)
    total_steps = 0
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _, _, _ = model(batch, sample=True)
            pred = pred[-1]
            batch_size = batch.num_graphs
            n_nodes = batch.n_nodes.tolist()
            pre_nodes = 0
            for i in range(batch_size):
                mol_labels.append(batch.rd_mol[i])
                mol_preds.append(set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]]))
                pre_nodes += n_nodes[i]
        with torch.no_grad():
            pred, extra_output, _, _ = model(batch, sample=True)
            if args.distributed:
                loss, loss_dict = model.module.compute_loss2(pred, extra_output, batch, args)
            else:
                loss, loss_dict = model.compute_loss2(pred, extra_output, batch, args)
            for k, v in loss_dict.items():
                loss_accum_dict[k] += v.item()
            pred = pred[-1]
            batch_size = batch.num_graphs
            n_nodes = batch.n_nodes.tolist()
            pre_nodes = 0
            for i in range(batch_size):
                mol_preds.append(set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]]))
                pre_nodes += n_nodes[i]
        total_steps += 1
    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= total_steps
    smiles2pairs = dict()
    for gen_mol in mol_preds:
        smiles = Chem.MolToSmiles(gen_mol)
        if smiles not in smiles2pairs:
            smiles2pairs[smiles] = [[gen_mol]]
        else:
            smiles2pairs[smiles][0].append(gen_mol)
    for ref_mol in mol_labels:
        smiles = Chem.MolToSmiles(ref_mol)
        if len(smiles2pairs[smiles]) == 1:
            smiles2pairs[smiles].append([ref_mol])
        else:
            smiles2pairs[smiles][1].append(ref_mol)
    del_smiles = []
    for smiles in smiles2pairs.keys():
        if len(smiles2pairs[smiles][1]) < 0 or len(smiles2pairs[smiles][1]) > 100000:
            del_smiles.append(smiles)
    for smiles in del_smiles:
        del smiles2pairs[smiles]
    cov_list = []
    mat_list = []
    pool = multiprocessing.Pool(args.num_workers)
    def input_args():
        for smiles in smiles2pairs.keys():
            yield smiles2pairs[smiles], args.use_ff, 0.5 if args.dataset_name == "qm9" else 1.25
    for res in tqdm(pool.imap(get_rmsd_min, input_args(), chunksize=10), total=len(smiles2pairs)):
        cov_list.append(res[0])
        mat_list.append(res[1])
    print(f"cov mean {np.mean(cov_list)} med {np.median(cov_list)}")
    print(f"mat mean {np.mean(mat_list)} med {np.median(mat_list)}")
    return np.mean(cov_list), np.mean(mat_list),loss_accum_dict
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint/NatGen/")
    parser.add_argument("--log_dir", type=str, default="./checkpoint/NatGen/NatGen-log/")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--use-adamw", action="store_true", default=True)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--base-path", type=str, default="dataset/qm9_processed")
    parser.add_argument("--dataset-name", type=str, choices=["qm9", "drugs", "iso17"], default="qm9")
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--lr-warmup", action="store_true", default=True)
    parser.add_argument("--enable-tb", action="store_true", default=True)
    parser.add_argument("--aux-loss", type=float, default=0.2)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--extend-edge", action="store_true", default=False)
    parser.add_argument("--data-split", type=str, choices=["cgcf", "default", "confgf"], default="confgf")
    parser.add_argument("--reuse-prior", action="store_true", default=True)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--vae-beta", type=float, default=1)
    parser.add_argument("--sample-beta", type=float, default=1.0)
    parser.add_argument("--vae-beta-max", type=float, default=0.03)#0.03
    parser.add_argument("--vae-beta-min", type=float, default=0.0001)#0.0001
    parser.add_argument("--pred-pos-residual", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--node-attn", action="store_true", default=True)
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
    parser.add_argument('--mixup', action='store_true', help='Whether to have Mixup')
    parser.add_argument('--mixup-ratio', type=float, default=0.5)
    parser.add_argument("--gradient-monitoring-interval", type=int, default=1, help="Interval for monitoring gradients to prevent explosion")
    parser.add_argument("--main-loss", type=float, default=0.5)
    parser.add_argument("--line-loss", type=float, default=0.25)
    parser.add_argument("--mixup-loss", type=float, default=0.25)
    args = parser.parse_args()
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
        batch_size=args.batch_size * 1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size * 1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size * 1,
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
        "mixup_ratio": args.mixup_ratio,
    }
    model = ConfGNN(batch_size=args.batch_size, **shared_params).to(device)
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
        print("Finetuning with partial pretrained weights.  GOOD LUCK!!!!!")
    new_layer_params = [p for n, p in model.named_parameters() if 'line_gnns' in n or 'line_prior_conf_pos' in n or 'mixup_conf_pos' in n]
    base_params = [p for n, p in model.named_parameters() if not ('line_gnns' in n or 'line_prior_conf_pos' in n or 'mixup_conf_pos' in n)]
    optimizer = optim.Adam([
        {'params': base_params, 'lr': args.lr * 1},
        {'params': new_layer_params, 'lr': args.lr * 1}
    ], lr=args.lr, weight_decay=args.weight_decay)
    restore_fn = os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
    if args.restore:
        if os.path.exists(restore_fn):
            print(f"Restore from {restore_fn}")
            restore_checkpoint = torch.load(restore_fn, map_location=torch.device(args.device))
            model_state_dict = restore_checkpoint["model_state_dict"]
            if args.distributed:
                model_state_dict = fix_key_name(model_state_dict, add_module=True)
            else:
                model_state_dict = fix_key_name(model_state_dict, add_module=False)
            model_without_ddp.load_state_dict(model_state_dict, strict=False)
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
        if args.distributed:
            sampler_train.set_epoch(epoch)
        CosineBeta.step(epoch - 1)
        print("=====Epoch {}".format(epoch))
        print("Training...")
        loss_dict, mixup_loss_dict, line_loss_dict= train(model, device, train_loader, optimizer, scheduler, args)#, line_loss_dict
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
            for k, v in mixup_loss_dict.items():
                tb_writer.add_scalar(f"mixup_training/{k}", v, epoch) 
            for k, v in line_loss_dict.items():
                tb_writer.add_scalar(f"line_training/{k}", v, epoch) 
        if epoch % 1 == 0:
            print("Evaluating Valid set")
            valid_cov, valid_mat, eval_valid_loss_dict = evaluate(model, device, valid_loader, args)
            print(f"Valid:  Cov: {valid_cov}, Mat: {valid_mat}")
            if args.log_dir:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logs = {
                    "Time": current_time,
                    "Epoch": epoch,
                    "Valid_Cov": valid_cov,  
                    "Valid_Mat": valid_mat,
                }
                with io.open(
                    os.path.join(args.log_dir, "log.txt"), "a", encoding="utf8", newline="\n"
                ) as tgt:
                    print(json.dumps(logs, default=handle_numpy), file=tgt)
            if args.enable_tb and tb_writer is not None:
                tb_writer.add_scalar("Cov/Valid", valid_cov, epoch)
                tb_writer.add_scalar("Mat/Valid", valid_mat, epoch)
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"training/{k}", v, epoch)
                for k, v in mixup_loss_dict.items():
                    tb_writer.add_scalar(f"mixup_training/{k}", v, epoch)
                for k, v in line_loss_dict.items():
                    tb_writer.add_scalar(f"line_training/{k}", v, epoch)
                for k, v in eval_valid_loss_dict.items():
                    tb_writer.add_scalar(f"evaluating/{k}", v, epoch)
    if args.log_dir and args.enable_tb:
        tb_writer.close()
    if args.distributed:
        torch.distributed.destroy_process_group()
    print("Finished training!")
if __name__ == "__main__":
    main()
