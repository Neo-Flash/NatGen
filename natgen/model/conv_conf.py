import torch
from torch import nn, Tensor
from typing import Optional
import math
import torch.nn.functional as F
from ..molecule.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
import time
import networkx as nx
class MetaLayer(nn.Module):
    def __init__(
        self,
        edge_model,
        node_model,
        global_model,
        aggregate_edges_for_node_fn=None,
        aggregate_edges_for_globals_fn=None,
        aggregate_nodes_for_globals_fn=None,
        node_attn=False,
        emb_dim=None,
        global_attn=False,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.aggregate_edges_for_node_fn = aggregate_edges_for_node_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        if node_attn:
            self.node_attn = NodeAttn(emb_dim, num_heads=None)
        else:
            self.node_attn = None
        if global_attn and self.global_model is not None:
            self.global_node_attn = GlobalAttn(emb_dim, num_heads=None)
            self.global_edge_attn = GlobalAttn(emb_dim, num_heads=None)
        else:
            self.global_node_attn = None
            self.global_edge_attn = None
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        edge_batch: Tensor,
        node_batch: Tensor,
        num_nodes: None,
        num_edges: None,
    ):
        row = edge_index[0]
        col = edge_index[1]
        if self.edge_model is not None:
            row = edge_index[0]
            col = edge_index[1]
            sent_attributes = x[row]
            received_attributes = x[col]
            global_edges = torch.repeat_interleave(u, num_edges, dim=0)
            feat_list = [edge_attr, sent_attributes, received_attributes, global_edges]#
            concat_feat = torch.cat(feat_list, dim=1)
            edge_attr = self.edge_model(concat_feat)
        if self.node_model is not None and self.node_attn is None:
            sent_attributes = self.aggregate_edges_for_node_fn(edge_attr, row, size=x.size(0))
            received_attributes = self.aggregate_edges_for_node_fn(edge_attr, col, size=x.size(0))
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]#
            x = self.node_model(torch.cat(feat_list, dim=1))
        elif self.node_model is not None:
            max_node_id = x.size(0)
            sent_attributes = self.node_attn(x[row], x[col], edge_attr, row, max_node_id)
            received_attributes = self.node_attn(x[col], x[row], edge_attr, col, max_node_id)
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]#
            x = self.node_model(torch.cat(feat_list, dim=1))
        if self.global_model is not None and self.global_node_attn is None:
            n_graph = u.size(0)
            node_attributes = self.aggregate_nodes_for_globals_fn(x, node_batch, size=n_graph)
            edge_attributes = self.aggregate_edges_for_globals_fn(
                edge_attr, edge_batch, size=n_graph
            )
            feat_list = [u, node_attributes, edge_attributes]
            u = self.global_model(torch.cat(feat_list, dim=-1))
        elif self.global_model is not None:
            n_graph = u.size(0)
            node_attributes = self.global_node_attn(
                torch.repeat_interleave(u, num_nodes, dim=0), x, node_batch, dim_size=n_graph
            )
            edge_attributes = self.global_edge_attn(
                torch.repeat_interleave(u, num_edges, dim=0),
                edge_attr,
                edge_batch,
                dim_size=n_graph,
            )
            feat_list = [u, node_attributes, edge_attributes]
            u = self.global_model(torch.cat(feat_list, dim=-1))
        return x, edge_attr, u
class DropoutIfTraining(nn.Module):
    """
    Borrow this implementation from deepmind
    """
    def __init__(self, p=0.0, submodule=None):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.submodule = submodule if submodule else nn.Identity()
    def forward(self, x):
        x = self.submodule(x)
        newones = x.new_ones((x.size(0), 1))
        newones = F.dropout(newones, p=self.p, training=self.training)
        out = x * newones
        return out
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(dropout))
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size
        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()
    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()
    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x
class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))
            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size
        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()
    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()
    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x
class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.m = nn.Linear(sum(get_atom_feature_dims()), emb_dim)
    def forward(self, x):
        return self.m(x)
class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.m = nn.Linear(sum(get_bond_feature_dims()), emb_dim)
    def forward(self, x):
        return self.m(x)
class GatedLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_c, dim_out)
    def forward(self, x, context):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        return self._layer(x) * gate + bias
class NodeAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0
        self.w1 = nn.Linear(3 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3 = nn.Linear(2 * emb_dim, emb_dim)
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))
    def forward(self, q, k_v, k_e, index, nnode):
        """
        q: [N, C]
        k: [N, 2*c]
        v: [N, 2*c]
        """
        x = torch.cat([q, k_v, k_e], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)
        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)
        return x
class GlobalAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0
        self.w1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros(self.num_heads, self.emb_dim // self.num_heads))
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
    def forward(self, q, k, index, dim_size):
        x = torch.cat([q, k], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)
        v = k.view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=dim_size)
        return x
def calculate_line_graph(x, edge_idx, edge_attr):
    start_time = time.time() 
    data = Data(edge_index=edge_idx, edge_attr=edge_attr)
    line_graph_transform = LineGraph()
    line_graph_data = line_graph_transform(data)
    line_graph_edge_index = line_graph_data.edge_index
    edges_a = edge_idx[:, line_graph_edge_index[0]]
    edges_b = edge_idx[:, line_graph_edge_index[1]]
    shared_nodes = edges_a[1] 
    node_b = edges_b[0]
    node_b[shared_nodes == edges_b[1]] = edges_b[1][shared_nodes == edges_b[1]]
    new_edge_attr = (x[edges_a[0]] + x[shared_nodes] + x[node_b]) / 3
    end_time = time.time() 
    return line_graph_edge_index, new_edge_attr
class LineLayer(nn.Module):
    def __init__(self, in_dim=256, dim=256, dropout=0.1, if_pos=False):
        super().__init__()
        self.dim = dim
        self.if_pos = if_pos
        self.linear = nn.Linear(in_dim, dim)
        self.linear2 = nn.Linear(in_dim * 2, dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Parameter(torch.randn(1, dim))
        if self.if_pos:
            num_gaussians = 6
            self.rbf_expand = RBFExpansion(0, 5, num_gaussians)
            self.linear_rbf = nn.Linear(num_gaussians, dim, bias=False)
        self.init_params()
        self.readout = ReadoutPhase(dim)
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        if self.if_pos:
            nn.init.xavier_uniform_(self.linear_rbf.weight)        
    def forward(self, x, edges, pos, batch,edge_idx, edge_attr ):
        start_time = time.time() 
        edge_idx, edge_attr = calculate_line_graph(x, edge_idx, edge_attr)
        x = self.dropout(x)
        x_src = self.linear(x).index_select(0, edges[:, 0])
        x_dst = self.linear(x).index_select(0, edges[:, 1])
        x = self.act(x_src + x_dst)
        if self.if_pos:
            pos_src = pos.index_select(0, edges[:, 0])
            pos_dst = pos.index_select(0, edges[:, 1])
            vector = pos_dst - pos_src
            distance = torch.norm(vector, p=2, dim=1).unsqueeze(-1)
            torch.clamp(distance, min=0.1)   
            distance_matrix = self.rbf_expand(distance)
            dist_emd = self.linear_rbf(distance_matrix)
            x = x * dist_emd
            pos = (pos_src + pos_dst) / 2
        atom_repr = x * self.attn
        atom_repr = nn.ELU()(atom_repr)
        batch = batch.index_select(0, edges[:, 0])
        mol_repr = self.readout(atom_repr, batch)
        mol_repr = self.linear2(mol_repr)
        end_time = time.time() 
        return atom_repr, pos, batch, mol_repr, edge_idx, edge_attr
class RBFExpansion(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(RBFExpansion, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)
    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
class ReadoutPhase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weighting = nn.Linear(dim, 1) 
        self.score = nn.Sigmoid() 
        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)
    def forward(self, x, batch):
        weighted = self.weighting(x)
        score = self.score(weighted)
        output1 = global_add_pool(score * x, batch)
        output2 = global_max_pool(x, batch)
        output = torch.cat([output1, output2], dim=1)
        return output