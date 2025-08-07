from genericpath import exists
import numpy as np
import random
import os
import json
from tqdm import tqdm
from random import sample
import pickle
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType
import torch
from natgen import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse import SparseTensor
import re
import natgen
from ..molecule.graph import rdk2graph, rdk2graphedge,rdk2graph_chiral
import copy
from rdkit.Chem.rdmolops import RemoveHs
from natgen.molecule.gt import isomorphic_core, isomorphic_core_from_graph
from natgen.model.gnn_chiral import one_hot_atoms_chiral, one_hot_bonds_chiral
from natgen.model.gnn_conf import one_hot_atoms, one_hot_bonds
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
import numpy as np
import warnings 
import time 
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
import rdkit
rdkit.rdBase.DisableLog('rdApp.warning')
rdkit.rdBase.DisableLog('rdApp.error')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
def calculate_neighborhood(edge_index, num_nodes):
    nei_src_index = []
    nei_tgt_index = [[] for _ in range(num_nodes)]
    for i, j in zip(edge_index[0], edge_index[1]):
        nei_src_index.append(i)
        nei_tgt_index[i].append(j)
    nei_src_index = np.unique(nei_src_index)
    max_nei = max(len(nei) for nei in nei_tgt_index)
    nei_tgt_mask = np.ones((max_nei, num_nodes), dtype=bool)
    for i, neis in enumerate(nei_tgt_index):
        nei_tgt_mask[:len(neis), i] = False
        nei_tgt_index[i].extend([-1] * (max_nei - len(neis)))  
    nei_tgt_index = np.array(nei_tgt_index)
    return nei_src_index, nei_tgt_index, nei_tgt_mask
def convert_to_line_graph_chiral(x, chiral, edge_index, edge_attr):
    x_dim = x.size(1)
    edge_attr_dim = edge_attr.size(1) 
    target_dim = 173
    if x_dim < target_dim:
        padded_x = torch.cat([x, torch.zeros(x.size(0), target_dim - x_dim)], dim=1)
    else:
        padded_x = x[:, :target_dim]
    if edge_attr_dim < target_dim:
        padded_edge_attr = torch.cat([edge_attr, torch.zeros(edge_attr.size(0), target_dim - edge_attr_dim)], dim=1)
    else:
        padded_edge_attr = edge_attr[:, :target_dim]
    line_graph_node_features = []
    line_graph_node_chirality = []
    line_graph_edges = []
    edge_to_node = {}
    for idx in range(edge_index.shape[1]):
        node_a, node_b = edge_index[:, idx]
        edge_feature = padded_edge_attr[idx]
        new_node_feature = (padded_x[node_a] + padded_x[node_b]) / 2 + edge_feature
        new_node_chirality = (chiral[node_a] + chiral[node_b]) / 2.0 
        line_graph_node_features.append(new_node_feature)
        line_graph_node_chirality.append(new_node_chirality)
        edge_to_node[(node_a.item(), node_b.item())] = idx
    for idx1 in range(edge_index.shape[1]):
        for idx2 in range(idx1 + 1, edge_index.shape[1]):
            node_a1, node_b1 = edge_index[:, idx1]
            node_a2, node_b2 = edge_index[:, idx2]
            if node_a1 in (node_a2.item(), node_b2.item()) or node_b1 in (node_a2.item(), node_b2.item()):
                line_graph_edges.append([edge_to_node[(node_a1.item(), node_b1.item())], edge_to_node[(node_a2.item(), node_b2.item())]])
    line_graph_node_features = torch.stack(line_graph_node_features)
    line_graph_node_chirality = torch.stack(line_graph_node_chirality)
    if len(line_graph_edges) > 0:
        line_graph_edge_index = torch.tensor(line_graph_edges).t().contiguous()
        line_graph_edge_attr = []
        for i in range(line_graph_edge_index.shape[1]):
            edge_a, edge_b = line_graph_edge_index[:, i]
            new_edge_feature = (padded_edge_attr[edge_a] + padded_edge_attr[edge_b]) / 2
            line_graph_edge_attr.append(new_edge_feature)
        line_graph_edge_attr = torch.stack(line_graph_edge_attr)
    else:
        line_graph_edge_index = torch.zeros((2, 0), dtype=torch.long)
        line_graph_edge_attr = torch.zeros((0, target_dim))
    line_graph_data = Data(x=line_graph_node_features, chirality=line_graph_node_chirality, edge_index=line_graph_edge_index, edge_attr=line_graph_edge_attr)
    return line_graph_data
def convert_to_line_graph(x, pos, edge_index, edge_attr):
    padded_edge_attr = torch.cat([edge_attr, torch.zeros(edge_attr.size(0), 173 - edge_attr.size(1))], dim=1)
    line_graph_node_features = []
    line_graph_node_pos = []
    line_graph_edges = []
    edge_to_node = {}
    for idx in range(edge_index.shape[1]):
        node_a, node_b = edge_index[:, idx]
        edge_feature = padded_edge_attr[idx]
        new_node_feature = (x[node_a] + x[node_b]) / 2 + edge_feature
        new_node_pos = (pos[node_a] + pos[node_b]) / 2
        line_graph_node_features.append(new_node_feature)
        line_graph_node_pos.append(new_node_pos)
        edge_to_node[(node_a.item(), node_b.item())] = idx
    for idx1 in range(edge_index.shape[1]):
        for idx2 in range(idx1 + 1, edge_index.shape[1]):
            node_a1, node_b1 = edge_index[:, idx1]
            node_a2, node_b2 = edge_index[:, idx2]
            if node_a1 in (node_a2.item(), node_b2.item()) or node_b1 in (node_a2.item(), node_b2.item()):
                line_graph_edges.append([edge_to_node[(node_a1.item(), node_b1.item())], edge_to_node[(node_a2.item(), node_b2.item())]])
    line_graph_node_features = torch.stack(line_graph_node_features)
    line_graph_node_pos = torch.stack(line_graph_node_pos)
    line_graph_edge_index = torch.tensor(line_graph_edges).t().contiguous()
    line_graph_edge_attr = []
    for i in range(line_graph_edge_index.shape[1]):
        edge_a, edge_b = line_graph_edge_index[:, i]
        new_edge_feature = (padded_edge_attr[edge_a] + padded_edge_attr[edge_b]) / 2
        line_graph_edge_attr.append(new_edge_feature)
    line_graph_edge_attr = torch.stack(line_graph_edge_attr)
    line_graph_data = Data(x=line_graph_node_features, pos=line_graph_node_pos, edge_index=line_graph_edge_index, edge_attr=line_graph_edge_attr)
    return line_graph_data
def create_batch(n_nodes):
    batch = [torch.full((n,), i, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) for i, n in enumerate(n_nodes)]
    return torch.cat(batch, dim=0)
def get_chiral_features0(mol):
    chiral_features = []
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    atom_chirality = {i: 0 for i in range(mol.GetNumAtoms())} 
    for idx, chirality in chiral_centers:
        if chirality == "?":
            return None 
        elif chirality == "R":
            atom_chirality[idx] = 1  
        elif chirality == "S":
            atom_chirality[idx] = 2 
    for i in range(mol.GetNumAtoms()):
        chiral_features.append(atom_chirality[i])
    return chiral_features
def get_chiral_features(mol):
    chiral_features = []
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    atom_chirality = {i: 0 for i in range(mol.GetNumAtoms())}
    for idx, chirality in chiral_centers:
        if chirality == "?":
            return None  
        elif chirality == "R":
            atom_chirality[idx] = 1  
        elif chirality == "S":
            atom_chirality[idx] = 2  
    if all(chirality == 0 for chirality in atom_chirality.values()):
        return None  
    for i in range(mol.GetNumAtoms()):
        chiral_features.append(atom_chirality[i])
    return chiral_features
def get_chiral_features2(mol):
    chiral_features = []
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    atom_chirality = {i: 0 for i in range(mol.GetNumAtoms())}
    for idx, chirality in chiral_centers:
        if chirality == "?":
            return None  
        elif chirality == "R":
            atom_chirality[idx] = 1 
        elif chirality == "S":
            atom_chirality[idx] = 2  
    for i in range(mol.GetNumAtoms()):
        chiral_features.append(atom_chirality[i])
    return chiral_features
def get_chiral_features3(mol):
    chiral_features = []
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    atom_chirality = {i: 0 for i in range(mol.GetNumAtoms())}
    for idx, chirality in chiral_centers:
        if chirality == "R":
            atom_chirality[idx] = 1 
        elif chirality == "S":
            atom_chirality[idx] = 2   
    for i in range(mol.GetNumAtoms()):
        chiral_features.append(atom_chirality[i])
    return chiral_features
class PygGeomDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        rdk2graph=rdk2graph,
        rdk2graph_chiral=rdk2graph_chiral,
        transform=None,
        pre_transform=None,
        dataset="qm9",
        base_path="/Users/flash/Desktop/dataset",
        seed=None,
        extend_edge=False,
        data_split="cgcf",
        remove_hs=False,
    ):
        self.original_root = root
        self.rdk2graph = rdk2graph
        self.rdk2graph_chiral = rdk2graph_chiral
        if seed == None:
            self.seed = 2021
        else:
            self.seed = seed
        assert dataset in ["qm9", "drugs", "iso17", "Coconut"]
        self.folder = os.path.join(root, f"geom_{dataset}_{data_split}")
        if extend_edge:
            self.rdk2graph = rdk2graphedge
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_ee")
        if remove_hs:
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_rh_ext_gt")
        else:
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_ext_gt")
        self.base_path = base_path
        self.dataset_name = dataset
        self.data_split = data_split
        self.remove_hs = remove_hs
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return "data.csv.gz"
    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"
    def download(self):
        if os.path.exists(self.processed_paths[0]):
            return
        else:
            assert os.path.exists(self.base_path)
    def process(self):
        assert self.dataset_name in ["qm9", "drugs", "iso17", "Coconut"]
        if self.data_split == "Coconut":
            self.process_Coconut(read_some=False)  
            return
        if self.data_split == "confgf":
            self.process_confgf()
            return
        summary_path = os.path.join(self.base_path, f"summary_{self.dataset_name}.json")
        with open(summary_path, "r") as src:
            summ = json.load(src)
        pickle_path_list = []
        for smiles, meta_mol in tqdm(summ.items()):
            u_conf = meta_mol.get("uniqueconfs")
            if u_conf is None:
                continue
            pickle_path = meta_mol.get("pickle_path")
            if pickle_path is None:
                continue
            if "." in smiles:
                continue
            pickle_path_list.append(pickle_path)
        data_list = []
        num_mols = 0
        num_confs = 0
        bad_case = 0
        random.seed(19970327)
        random.shuffle(pickle_path_list)
        train_size = int(len(pickle_path_list) * 0.8)
        valid_size = int(len(pickle_path_list) * 0.1)
        train_idx = []
        valid_idx = []
        test_idx = []
        for i, pickle_path in enumerate(tqdm(pickle_path_list)):
            if self.dataset_name in ["drugs"]:
                if i < train_size:
                    if len(train_idx) >= 2000000:
                        continue
                elif i < valid_size:
                    if len(valid_idx) >= 100000:
                        continue
                else:
                    if len(test_idx) >= 100000:
                        continue
            with open(os.path.join(self.base_path, pickle_path), "rb") as src:
                mol = pickle.load(src)
            if mol.get("uniqueconfs") != len(mol.get("conformers")):
                bad_case += 1
                continue
            if mol.get("uniqueconfs") <= 0:
                bad_case += 1
                continue
            if mol.get("conformers")[0]["rd_mol"].GetNumBonds() < 1:
                bad_case += 1
                continue
            if "." in Chem.MolToSmiles(mol.get("conformers")[0]["rd_mol"]):
                bad_case += 1
                continue
            num_mols += 1
            for conf_meta in mol.get("conformers"):
                if self.remove_hs:
                    try:
                        new_mol = RemoveHs(conf_meta["rd_mol"])
                    except Exception:
                        continue
                else:
                    new_mol = conf_meta["rd_mol"]
                graph = self.rdk2graph(new_mol)
                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]
                data = Data()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(new_mol.GetConformer(0).GetPositions()).to(torch.float)
                data.lowestenergy = torch.as_tensor([mol.get("lowestenergy")]).to(torch.float)
                data.energy = torch.as_tensor([conf_meta["totalenergy"]]).to(torch.float)
                data.rd_mol = copy.deepcopy(new_mol)
                data.isomorphisms = isomorphic_core(new_mol)
                data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
                data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
                data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
                if i < train_size:
                    train_idx.append(len(data_list))
                elif i < valid_size:
                    valid_idx.append(len(data_list))
                else:
                    test_idx.append(len(data_list))
                data_list.append(data)
                num_confs += 1
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num mols {num_mols} num confs {num_confs} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )
    def get_idx_split(self):
        path = os.path.join(self.root, "split")
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))
    def process_confgf(self):
        valid_conformation = 0
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        bad_case = 0
        file_name = ["train_data_40k", "val_data_5k", "test_data_200"]
        if self.dataset_name == "drugs":
            file_name[0] = "train_data_39k"
        print("Converting pickle files into graphs...")
        for subset in file_name:
            pkl_fn = os.path.join(self.base_path, f"{subset}.pkl")
            with open(pkl_fn, "rb") as src:
                mol_list = pickle.load(src)
            mol_list = [x.rdmol for x in mol_list]
            for mol in tqdm(mol_list):
                if self.remove_hs:
                    try:
                        mol = RemoveHs(mol)
                    except Exception:
                        continue
                if "." in Chem.MolToSmiles(mol):
                    bad_case += 1
                    continue
                if mol.GetNumBonds() < 1:
                    bad_case += 1
                    continue
                graph = self.rdk2graph(mol)
                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]
                data = CustomData()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.full_edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = one_hot_bonds(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
                data.x = one_hot_atoms(torch.from_numpy(graph["node_feat"])).to(torch.float32)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
                data.rd_mol = copy.deepcopy(mol)
                data.isomorphisms = isomorphic_core(mol)
                data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
                data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
                data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
                line_graph = convert_to_line_graph(data.x, data.pos, data.edge_index, data.edge_attr)
                data.line_graph_x = line_graph.x
                data.line_graph_edge_attr = line_graph.edge_attr
                data.line_graph_edge_index = line_graph.edge_index
                data.line_graph_n_nodes = len(line_graph.x)
                data.line_graph_n_edges = line_graph.edge_index.shape[1] 
                data.line_graph_pos  =  line_graph.pos
                data.line_graph_batch = create_batch([data.line_graph_n_nodes])
                data.line_graph_isomorphisms = isomorphic_core_from_graph(torch.tensor(line_graph.x).to(torch.float32), torch.tensor(line_graph.edge_index).to(torch.int64), torch.tensor(line_graph.edge_attr).to(torch.float32))
                if "train" in subset:
                    train_idx.append(valid_conformation)
                elif "val" in subset:
                    valid_idx.append(valid_conformation)
                else:
                    test_idx.append(valid_conformation)
                valid_conformation += 1
                data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        if len(valid_idx) == 0:
            valid_idx = train_idx[:6400]
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )
    def process_Coconut(self, read_some=False):
        print("Loading SMILES from separate train/valid/test CSV files...")
        train_df = pd.read_csv("./dataset/train.csv")
        valid_df = pd.read_csv("./dataset/valid.csv")
        test_df = pd.read_csv("./dataset/test.csv")
        print(f"Loading full datasets - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        print("🚀 FULL TRAINING MODE: Using 100% of the data...")
        train_smiles = train_df['SMILES'].tolist()
        valid_smiles = valid_df['SMILES'].tolist()
        test_smiles = test_df['SMILES'].tolist()
        print(f"Loaded {len(train_smiles)} training, {len(valid_smiles)} validation, {len(test_smiles)} test SMILES")
        def process_smiles_batch(smiles_list, batch_name):
            print(f"Converting {batch_name} SMILES to molecules...")
            mol_list = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list, desc=f"{batch_name} SMILES conversion")]
            mol_list = [mol for mol in tqdm(mol_list, desc=f"Filtering valid {batch_name} molecules") if mol is not None]
            return mol_list
        train_mols = process_smiles_batch(train_smiles, "training")
        valid_mols = process_smiles_batch(valid_smiles, "validation")
        test_mols = process_smiles_batch(test_smiles, "test")
        valid_conformation = 0
        bad_case = 0
        data_list = []
        train_idx, valid_idx, test_idx = [], [], []
        print("Processing training molecules...")
        for mol in tqdm(train_mols, desc="Processing training molecules"):
            chiral_features = get_chiral_features3(mol)
            if chiral_features is None:
                bad_case += 1
                continue
            if self.remove_hs:
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    bad_case += 1
                    continue
            if "." in Chem.MolToSmiles(mol) or mol.GetNumBonds() < 1:
                bad_case += 1
                continue
            try:
                graph = self.rdk2graph_chiral(mol)
            except Exception as e:
                print(f"Skipping training molecule due to error: {e}")
                bad_case += 1
                continue
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = one_hot_bonds_chiral(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
            data.x = one_hot_atoms_chiral(torch.from_numpy(graph["node_feat"])).to(torch.float32)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.rd_mol = copy.deepcopy(mol)
            data.chiral = torch.tensor(chiral_features, dtype=torch.int64)
            assert len(data.chiral) == data.x.size(0), "Chirality tensor length mismatch"
            line_graph = convert_to_line_graph_chiral(data.x, data.chiral, data.edge_index, data.edge_attr)
            data.line_graph_x = line_graph.x
            data.line_graph_edge_attr = line_graph.edge_attr
            data.line_graph_edge_index = line_graph.edge_index
            data.line_graph_n_nodes = len(line_graph.x)
            data.line_graph_n_edges = line_graph.edge_index.shape[1]
            data.line_graph_chirality = line_graph.chirality
            data.line_graph_batch = create_batch([data.line_graph_n_nodes])
            train_idx.append(len(data_list))
            data_list.append(data)
            valid_conformation += 1
        print("Processing validation molecules...")
        for mol in tqdm(valid_mols, desc="Processing validation molecules"):
            chiral_features = get_chiral_features3(mol)
            if chiral_features is None:
                bad_case += 1
                continue
            if self.remove_hs:
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    bad_case += 1
                    continue
            if "." in Chem.MolToSmiles(mol) or mol.GetNumBonds() < 1:
                bad_case += 1
                continue
            try:
                graph = self.rdk2graph_chiral(mol)
            except Exception as e:
                print(f"Skipping validation molecule due to error: {e}")
                bad_case += 1
                continue
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = one_hot_bonds_chiral(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
            data.x = one_hot_atoms_chiral(torch.from_numpy(graph["node_feat"])).to(torch.float32)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.rd_mol = copy.deepcopy(mol)
            data.chiral = torch.tensor(chiral_features, dtype=torch.int64)
            assert len(data.chiral) == data.x.size(0), "Chirality tensor length mismatch"
            line_graph = convert_to_line_graph_chiral(data.x, data.chiral, data.edge_index, data.edge_attr)
            data.line_graph_x = line_graph.x
            data.line_graph_edge_attr = line_graph.edge_attr
            data.line_graph_edge_index = line_graph.edge_index
            data.line_graph_n_nodes = len(line_graph.x)
            data.line_graph_n_edges = line_graph.edge_index.shape[1]
            data.line_graph_chirality = line_graph.chirality
            data.line_graph_batch = create_batch([data.line_graph_n_nodes])
            valid_idx.append(len(data_list))
            data_list.append(data)
            valid_conformation += 1
        print("Processing test molecules...")
        for mol in tqdm(test_mols, desc="Processing test molecules"):
            chiral_features = get_chiral_features3(mol)
            if chiral_features is None:
                bad_case += 1
                continue
            if self.remove_hs:
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    bad_case += 1
                    continue
            if "." in Chem.MolToSmiles(mol) or mol.GetNumBonds() < 1:
                bad_case += 1
                continue
            try:
                graph = self.rdk2graph_chiral(mol)
            except Exception as e:
                print(f"Skipping test molecule due to error: {e}")
                bad_case += 1
                continue
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = one_hot_bonds_chiral(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
            data.x = one_hot_atoms_chiral(torch.from_numpy(graph["node_feat"])).to(torch.float32)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.rd_mol = copy.deepcopy(mol)
            data.chiral = torch.tensor(chiral_features, dtype=torch.int64)
            assert len(data.chiral) == data.x.size(0), "Chirality tensor length mismatch"
            line_graph = convert_to_line_graph_chiral(data.x, data.chiral, data.edge_index, data.edge_attr)
            data.line_graph_x = line_graph.x
            data.line_graph_edge_attr = line_graph.edge_attr
            data.line_graph_edge_index = line_graph.edge_index
            data.line_graph_n_nodes = len(line_graph.x)
            data.line_graph_n_edges = line_graph.edge_index.shape[1]
            data.line_graph_chirality = line_graph.chirality
            data.line_graph_batch = create_batch([data.line_graph_n_nodes])
            test_idx.append(len(data_list))
            data_list.append(data)
            valid_conformation += 1
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print(f"Saved {valid_conformation} conformations, {bad_case} bad cases")
        print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
        torch.save((data, slices), self.processed_paths[0])
        split_dict = {
            "train": torch.tensor(train_idx, dtype=torch.long),
            "valid": torch.tensor(valid_idx, dtype=torch.long),
            "test": torch.tensor(test_idx, dtype=torch.long),
        }
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(split_dict, os.path.join(self.root, "split", "split_dict.pt"))
class CustomData(Data):
    def __cat_dim__(self, key, value, *args):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask)", key)):
            return -1
        return 0
