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
    init_distributed_mode,
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
import time
from torch.utils.data import Subset
from rdkit.Chem import SDWriter
from rdkit.Chem import Descriptors
import csv
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
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _ , _, _   = model(batch, sample=True)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_labels.append(batch.rd_mol[i])
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]
        with torch.no_grad():
            pred, _ , _, _   = model(batch, sample=True)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]
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
        if len(smiles2pairs[smiles][1]) < 0 or len(smiles2pairs[smiles][1]) > 10000:
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
    return np.mean(cov_list), np.mean(mat_list)
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0002)
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
    parser.add_argument("--eval-from", type=str, default="checkpoint/checkpoint.pt")
    parser.add_argument("--extend-edge", action="store_true", default=False)
    parser.add_argument("--data-split", type=str, choices=["cgcf", "default", "confgf"], default="confgf")
    parser.add_argument("--reuse-prior", action="store_true", default=True)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--vae-beta", type=float, default=1.0)
    parser.add_argument("--sample-beta", type=float, default=1)
    parser.add_argument("--vae-beta-max", type=float, default=0.03)
    parser.add_argument("--vae-beta-min", type=float, default=0.0001)
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
    parser.add_argument("--gradient-monitoring-interval", type=int, default=1, help="Interval for monitoring gradients to prevent explosion")
    args = parser.parse_args()
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
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
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
        "sample_beta": args.sample_beta,
        "rand_aug": args.rand_aug,
        "no_3drot": args.no_3drot,
        "not_origin": args.not_origin,
        "sample_beta": args.sample_beta,
    }
    model = ConfGNN(**shared_params).to(device)
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
    print("Evaluating...")
    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")
    test_cov, test_mat = evaluate(model, device, test_loader, args)
if __name__ == "__main__":
    main()
