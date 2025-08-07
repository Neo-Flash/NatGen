import argparse
import torch
from torch_geometric.loader import DataLoader
from natgen.e2c.dataset import PygGeomDataset
from natgen.model.gnn_chiral import ChiralGNN
from natgen.utils.utils import set_rdmol_positions, get_best_rmsd, evaluate_distance
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import torch
import argparse
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from natgen.e2c.dataset import PygGeomDataset, CustomData, get_chiral_features3
from natgen.model.gnn_chiral import ChiralGNN
from natgen.utils.utils import set_rdmol_positions, get_best_rmsd, evaluate_distance
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
import copy
import pandas as pd
from natgen.molecule.graph import rdk2graph_chiral
from natgen.model.gnn_chiral import one_hot_atoms_chiral, one_hot_bonds_chiral
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import csv
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
            chirality_list, extra_output,_,_ = model(batch, sample=True)
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint/NatGen/checkpoint.pt")
    parser.add_argument("--base-path", type=str, default="./dataset/")
    parser.add_argument("--dataset-name", type=str, choices=["Coconut"], default="Coconut")
    parser.add_argument("--data-split", type=str, choices=["Coconut"], default="Coconut")
    parser.add_argument("--dropedge-rate", type=float, default=0.15)
    parser.add_argument("--dropnode-rate", type=float, default=0.15)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--mlp-hidden-size", type=int, default=256)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--encoder-dropout", type=float, default=0.15)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=True)
    parser.add_argument("--vae-beta", type=float, default=1)
    parser.add_argument("--reuse-prior", action="store_true", default=True)
    parser.add_argument("--sample-beta", type=float, default=1.0)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--pred-pos-residual", action="store_true", default=True)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--shared-decoder", action="store_true", default=False)
    parser.add_argument("--shared-output", action="store_true", default=True)
    parser.add_argument("--use-global", action="store_true", default=False)
    parser.add_argument("--sg-pos", action="store_true", default=False)
    parser.add_argument("--use-ss", action="store_true", default=False)
    parser.add_argument("--rand-aug", action="store_true", default=False)
    parser.add_argument("--no-3drot", action="store_true", default=True)
    parser.add_argument("--not-origin", action="store_true", default=False)
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = PygGeomDataset(
        root="dataset",
        dataset=args.dataset_name,
        base_path=args.base_path,
        seed=2021,
        extend_edge=False,
        data_split=args.data_split,
        remove_hs=True,
    )
    split_idx = dataset.get_idx_split()
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
    train_loader = DataLoader(
        dataset[split_idx["train"]],
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
        "global_reducer": "sum",
        "node_reducer": "sum",
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
    model = ChiralGNN(batch_size=args.batch_size, **shared_params).to(device)
    checkpoint_path = args.checkpoint_dir
    assert os.path.exists(checkpoint_path), "Checkpoint not found"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Model loaded successfully.")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    print("Evaluating model on test set...")
    chirality_accuracy_chiral, _, _ = evaluate(model, device, test_loader, args)
    print(f"100% chirality_accuracy_chiral: {chirality_accuracy_chiral:.4f}")
if __name__ == "__main__":
    main()