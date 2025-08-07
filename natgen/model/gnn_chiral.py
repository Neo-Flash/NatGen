import torch
from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from ..molecule.features import get_atom_feature_dims_chiral, get_bond_feature_dims
from .conv_chiral import DropoutIfTraining, MLP, MetaLayer, MLPwoLastAct
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
import time
class SoftMixupProcessor(nn.Module):
    def __init__(self, latent_size, dropout=0.1):
        super().__init__()
        self.inter_aligner = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.LayerNorm(latent_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.intra_aligner = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.LayerNorm(latent_size), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(latent_size * 2, latent_size),
            nn.Sigmoid()
        )
    def forward(self, original_feat, mixup_feat, mode):
        if mode == 'inter':
            aligned_feat = self.inter_aligner(mixup_feat)
        elif mode == 'intra':
            aligned_feat = self.intra_aligner(mixup_feat)
        else:
            return mixup_feat
        gate_input = torch.cat([original_feat, aligned_feat], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused_feat = gate * original_feat + (1 - gate) * aligned_feat
        return fused_feat
class ResidualMixupAdapter(nn.Module):
    def __init__(self, latent_size, num_blocks=2):
        super().__init__()
        self.adaptation_blocks = nn.ModuleList([
            self._make_adapter_block(latent_size) for _ in range(num_blocks)
        ])
    def _make_adapter_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 4), 
            nn.ReLU(),
            nn.Linear(dim // 4, dim),  
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        for block in self.adaptation_blocks:
            x = x + block(x)  
        return x
def calculate_edges_per_graph_undirected_modified(edge_index, batch):
    num_graphs = batch.num_graphs
    edges_per_graph = [0] * num_graphs
    if not hasattr(batch, 'batch'):
        raise AttributeError("batch error")
    for i in range(edge_index.shape[1]):
        src, dest = edge_index[:, i]
        graph_id = batch.batch[src].item()  
        edges_per_graph[graph_id] += 1
    return torch.tensor(edges_per_graph)
_REDUCER_NAMES = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}
class ChiralGNN(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 256,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 12,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        dropout: float = 0.1,
        graph_pooling: str = "sum",
        layernorm_before: bool = False,
        pooler_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        use_bn: bool = False,
        vae_beta: float = 1.0,
        decoder_layers: int = None,
        reuse_prior: bool = False,
        cycle: int = 1,
        pred_pos_residual: bool = False,
        node_attn: bool = False,
        shared_decoder: bool = False,
        shared_output: bool = False,
        global_attn: bool = False,
        sample_beta: float = 1,
        use_global: bool = False,
        sg_pos: bool = False,
        use_ss: bool = False,
        rand_aug: bool = False,
        no_3drot: bool = False,
        not_origin: bool = False,
        batch_size: int = 2,
        inter_mixup_weight: float = 0.0,
        intra_mixup_weight: float = 0.0,
        mixup_ratio: float = 0.5,
    ):
        super().__init__()
        self.encoder_edge = MLP(
            sum(get_bond_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.encoder_node = MLP(
            sum(get_atom_feature_dims_chiral()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.encoder_line_edge = MLP(
            173,
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.encoder_line_node = MLP(
            173,
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.global_init = nn.Parameter(torch.zeros((1, latent_size)))
        self.chiral_embedding = MLP(3, [latent_size // 4, latent_size])
        self.dis_embedding = MLP(1, [latent_size // 4, latent_size])
        if num_message_passing_steps == 1:
            num_shared_layers = 1
            num_specific_layers = 1 
        elif num_message_passing_steps == 2:
            num_shared_layers = 1 
            num_specific_layers = 1
        else:
            num_shared_layers = max(1, int(num_message_passing_steps * 0.1))
            num_specific_layers = max(1, num_message_passing_steps - num_shared_layers)
            if num_shared_layers + num_specific_layers != num_message_passing_steps:
                num_shared_layers = max(1, num_message_passing_steps - 1)
                num_specific_layers = num_message_passing_steps - num_shared_layers
        self.shared_prior_gnns = nn.ModuleList()
        self.shared_prior_pos = nn.ModuleList()
        for _ in range(num_shared_layers):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            global_model = MLP(
                latent_size * 3,
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                layernorm_before=layernorm_before,
                dropout=encoder_dropout,
                use_bn=use_bn,
            )
            self.shared_prior_gnns.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
            self.shared_prior_pos.append(MLPwoLastAct(latent_size, [latent_size, 3]))
        self.original_prior_gnns = nn.ModuleList()
        self.original_prior_pos = nn.ModuleList()
        for _ in range(num_message_passing_steps): 
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            global_model = MLP(
                latent_size * 3,
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                layernorm_before=layernorm_before,
                dropout=encoder_dropout,
                use_bn=use_bn,
            )
            self.original_prior_gnns.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
            self.original_prior_pos.append(MLPwoLastAct(latent_size, [latent_size, 3]))
        self.inter_prior_gnns = nn.ModuleList() 
        self.inter_prior_pos = nn.ModuleList()
        self.intra_prior_gnns = nn.ModuleList()
        self.intra_prior_pos = nn.ModuleList()
        for mode in ['inter', 'intra']:
            for _ in range(num_specific_layers):
                edge_model = DropoutIfTraining(
                    p=dropedge_rate,
                    submodule=MLP(
                        latent_size * 4,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    ),
                )
                node_model = DropoutIfTraining(
                    p=dropnode_rate,
                    submodule=MLP(
                        latent_size * 4,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    ),
                )
                global_model = MLP(
                    latent_size * 3,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
                meta_layer = MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
                pos_layer = MLPwoLastAct(latent_size, [latent_size, 3])
                if mode == 'inter':
                    self.inter_prior_gnns.append(meta_layer)
                    self.inter_prior_pos.append(pos_layer)
                else:  
                    self.intra_prior_gnns.append(meta_layer) 
                    self.intra_prior_pos.append(pos_layer)
        self.soft_mixup_processor = SoftMixupProcessor(latent_size, dropout)
        self.residual_adapter = ResidualMixupAdapter(latent_size)
        self.num_message_passing_steps = num_message_passing_steps
        self.num_shared_layers = num_shared_layers
        self.num_specific_layers = num_specific_layers
        self.encoder_gnns = nn.ModuleList()
        for i in range(1):  
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * 4,
                    [16] * (1) + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * 4,
                    [16] * (1) + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1 and not use_global:
                global_model = None
            else:
                global_model = MLP(
                    latent_size * 3,
                    [16] * (1) + [latent_size], 
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
            self.encoder_gnns.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
        self.encoder_head = MLPwoLastAct(latent_size, [latent_size, 2 * latent_size], use_bn=use_bn)
        self.decoder_gnns = nn.ModuleList()
        self.decoder_pos = nn.ModuleList()
        decoder_layers = num_message_passing_steps if decoder_layers is None else decoder_layers
        for i in range(1): 
            if (not shared_decoder) or i == 0:
                edge_model = DropoutIfTraining(
                    p=dropedge_rate,
                    submodule=MLP(
                        latent_size * 4,
                        [16] * (1) + [latent_size], 
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    ),
                )
                node_model = DropoutIfTraining(
                    p=dropnode_rate,
                    submodule=MLP(
                        latent_size * 4,
                        [16] * (1) + [latent_size],  
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    ),
                )
                if i == decoder_layers - 1:
                    global_model = None
                else:
                    global_model = MLP(
                        latent_size * 3,
                        [16] * (1) + [latent_size], 
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    )
                self.decoder_gnns.append(
                    MetaLayer(
                        edge_model=edge_model,
                        node_model=node_model,
                        global_model=global_model,
                        aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                        aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                        aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                        node_attn=node_attn,
                        emb_dim=latent_size,
                        global_attn=global_attn,
                    )
                )
                if (not shared_output) or i == 0:
                    self.decoder_pos.append(MLPwoLastAct(latent_size, [latent_size, 3]))
                else:
                    self.decoder_pos.append(self.decoder_pos[-1])
            else:
                self.decoder_gnns.append(self.decoder_gnns[-1])
                self.decoder_pos.append(self.decoder_pos[-1])
        self.pooling = _REDUCER_NAMES[graph_pooling]
        self.pos_embedding = MLP(3, [latent_size, latent_size])
        self.pos_embedding2 = MLP(1, [latent_size, latent_size])
        self.dis_embedding = MLP(1, [latent_size, latent_size])
        self.latent_size = latent_size
        self.dropout = dropout
        self.vae_beta = vae_beta
        self.reuse_prior = reuse_prior
        self.cycle = cycle
        self.pred_pos_residual = pred_pos_residual
        self.sample_beta = sample_beta
        self.use_global = use_global
        self.sg_pos = sg_pos
        self.not_origin = not_origin
        self.use_ss = use_ss
        self.batch_size = batch_size
        self.rand_aug = rand_aug
        self.no_3drot = no_3drot
        self.inter_mixup_weight = inter_mixup_weight
        self.intra_mixup_weight = intra_mixup_weight
        self.mixup_ratio = mixup_ratio
        if self.use_ss:
            self.projection_head = MLPwoLastAct(
                latent_size, [mlp_hidden_size, latent_size], use_bn=True
            )
            self.prediction_head = MLPwoLastAct(
                latent_size, [mlp_hidden_size, latent_size], use_bn=True
            )
    def _forward_shared_then_specific(self, x, edge_attr, u, edge_index, edge_batch, 
                                     node_batch, num_nodes, num_edges, mode='original'):
        chirality_list = []
        original_x = x.clone() 
        for i, layer in enumerate(self.shared_prior_gnns):
            extended_x, extended_edge_attr = x, edge_attr
            x_1, edge_attr_1, u_1 = layer(
                extended_x, edge_index, extended_edge_attr, u,
                edge_batch, node_batch, num_nodes, num_edges
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            cur_chirality = self.shared_prior_pos[i](x)
            chirality_list.append(cur_chirality)
        if mode == 'original':
            specific_gnns = self.original_prior_gnns
            specific_pos = self.original_prior_pos
        elif mode == 'inter':
            specific_gnns = self.inter_prior_gnns 
            specific_pos = self.inter_prior_pos
        else: 
            specific_gnns = self.intra_prior_gnns
            specific_pos = self.intra_prior_pos
        for i, layer in enumerate(specific_gnns):
            extended_x, extended_edge_attr = x, edge_attr
            x_1, edge_attr_1, u_1 = layer(
                extended_x, edge_index, extended_edge_attr, u,
                edge_batch, node_batch, num_nodes, num_edges
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr  
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            cur_chirality = specific_pos[i](x)
            chirality_list.append(cur_chirality)
        if mode != 'original':
            x = self.soft_mixup_processor(original_x, x, mode)
            x = self.residual_adapter(x)
            final_chirality = specific_pos[-1](x)
            chirality_list[-1] = final_chirality
        return x, edge_attr, u, chirality_list
    def forward(self, batch, sample=False):
        need_mixup = self.training and (self.inter_mixup_weight > 0 or self.intra_mixup_weight > 0)
        if not need_mixup:
            return self._forward_original_with_shared_arch(batch, sample)
        else:
            return self._forward_with_mixup(batch, sample)
    def _forward_original_with_shared_arch(self, batch, sample=False):
        (x, edge_index, edge_attr, node_batch, num_nodes, num_edges, num_graphs,) = (
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.n_nodes, batch.n_edges, batch.num_graphs,
        )
        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)
        x_embed = self.encoder_node(x)
        edge_attr_embed = self.encoder_edge(edge_attr)
        u_embed = self.global_init.expand(num_graphs, -1)
        chirality_list = []
        x = x_embed
        edge_attr = edge_attr_embed
        u = u_embed
        extra_output = {}
        for i, layer in enumerate(self.original_prior_gnns): 
            extended_x, extended_edge_attr = x, edge_attr
            x_1, edge_attr_1, u_1 = layer(
                extended_x, edge_index, extended_edge_attr, u_embed,
                edge_batch, node_batch, num_nodes, num_edges,
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            cur_chirality = self.original_prior_pos[i](x)
            chirality_list.append(cur_chirality)
        extra_output["prior_chiral_list"] = chirality_list[-1]
        prior_output = [x, edge_attr, u]
        if not sample:
            x = x_embed
            edge_attr = edge_attr_embed
            u = u_embed
            cur_chirality = batch.chiral.unsqueeze(1) 
            for i, layer in enumerate(self.encoder_gnns):
                extended_x, extended_edge_attr = x, edge_attr
                x_1, edge_attr_1, u_1 = layer(
                    extended_x, edge_index, extended_edge_attr, u,
                    edge_batch, node_batch, num_nodes, num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            if self.use_global:
                aggregated_feat = u
            else:
                aggregated_feat = self.pooling(x, node_batch)
            latent = self.encoder_head(aggregated_feat)
            latent_mean, latent_logstd = torch.chunk(latent, chunks=2, dim=-1)
            extra_output["latent_mean"] = latent_mean
            extra_output["latent_logstd"] = latent_logstd
            z = self.reparameterization(latent_mean, latent_logstd)
        else:
            z = torch.randn_like(u_embed) * self.sample_beta
        z = torch.repeat_interleave(z, num_nodes, dim=0)
        if self.reuse_prior:
            x, edge_attr, u = prior_output
        else:
            x, edge_attr, u = x_embed, edge_attr_embed, u_embed
        cur_chirality = chirality_list[-1]
        chirality_list = []
        for i, layer in enumerate(self.decoder_gnns):
            if i == len(self.decoder_gnns) - 1:
                cycle = self.cycle
            else:
                cycle = 1
            for _ in range(cycle):
                extended_x, extended_edge_attr = x + z, edge_attr
                x_1, edge_attr_1, u_1 = layer(
                    extended_x, edge_index, extended_edge_attr, u,
                    edge_batch, node_batch, num_nodes, num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                cur_chirality = self.decoder_pos[i](x)
                chirality_list.append(cur_chirality)
        return chirality_list, extra_output, None, None
    def _forward_with_mixup(self, batch, sample=False):
        (x, edge_index, edge_attr, node_batch, num_nodes, num_edges, num_graphs,) = (
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.n_nodes, batch.n_edges, batch.num_graphs,
        )
        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)
        x_embed = self.encoder_node(x)
        edge_attr_embed = self.encoder_edge(edge_attr)
        u_embed = self.global_init.expand(num_graphs, -1)
        chirality_list = []
        x = x_embed
        edge_attr = edge_attr_embed
        u = u_embed
        extra_output = {}
        for i, layer in enumerate(self.original_prior_gnns):
            extended_x, extended_edge_attr = x, edge_attr
            x_1, edge_attr_1, u_1 = layer(
                extended_x, edge_index, extended_edge_attr, u_embed,
                edge_batch, node_batch, num_nodes, num_edges,
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            cur_chirality = self.original_prior_pos[i](x)
            chirality_list.append(cur_chirality)
        extra_output["prior_chiral_list"] = chirality_list[-1]
        prior_output = [x, edge_attr, u]
        if not sample:
            x = x_embed
            edge_attr = edge_attr_embed
            u = u_embed
            cur_chirality = batch.chiral.unsqueeze(1) 
            for i, layer in enumerate(self.encoder_gnns):
                extended_x, extended_edge_attr = x, edge_attr
                x_1, edge_attr_1, u_1 = layer(
                    extended_x, edge_index, extended_edge_attr, u,
                    edge_batch, node_batch, num_nodes, num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            if self.use_global:
                aggregated_feat = u
            else:
                aggregated_feat = self.pooling(x, node_batch)
            latent = self.encoder_head(aggregated_feat)
            latent_mean, latent_logstd = torch.chunk(latent, chunks=2, dim=-1)
            extra_output["latent_mean"] = latent_mean
            extra_output["latent_logstd"] = latent_logstd
            z = self.reparameterization(latent_mean, latent_logstd)
        else:
            z = torch.randn_like(u_embed) * self.sample_beta
        z = torch.repeat_interleave(z, num_nodes, dim=0)
        if self.reuse_prior:
            x, edge_attr, u = prior_output
        else:
            x, edge_attr, u = x_embed, edge_attr_embed, u_embed
        cur_chirality = chirality_list[-1]
        chirality_list = []
        for i, layer in enumerate(self.decoder_gnns):
            if i == len(self.decoder_gnns) - 1:
                cycle = self.cycle
            else:
                cycle = 1
            for _ in range(cycle):
                extended_x, extended_edge_attr = x + z, edge_attr
                x_1, edge_attr_1, u_1 = layer(
                    extended_x, edge_index, extended_edge_attr, u,
                    edge_batch, node_batch, num_nodes, num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                cur_chirality = self.decoder_pos[i](x)
                chirality_list.append(cur_chirality)
        inter_mixup_logits = None
        intra_mixup_logits = None
        if self.inter_mixup_weight > 0:
            inter_mixup_logits = self._compute_inter_mixup_enhanced(
                batch, x_embed, edge_attr_embed, u_embed, 
                edge_index, edge_batch, node_batch, num_nodes, num_edges
            )
        if self.intra_mixup_weight > 0:
            intra_mixup_logits = self._compute_intra_mixup_enhanced(
                batch, x_embed, edge_attr_embed, u_embed,
                edge_index, edge_batch, node_batch, num_nodes, num_edges
            )
        return chirality_list, extra_output, inter_mixup_logits, intra_mixup_logits
    def _compute_inter_mixup_enhanced(self, batch, x_embed, edge_attr_embed, u_embed, 
                                     edge_index, edge_batch, node_batch, num_nodes, num_edges):
        x, edge_attr, u = x_embed, edge_attr_embed, u_embed  
        lambda_ = self.mixup_ratio
        graph_idx = torch.arange(batch.num_graphs).to(x.device)
        index = torch.randperm(batch.x.shape[0])
        index_new = torch.zeros(batch.x.shape[0], dtype=torch.long)
        index_new[index] = torch.arange(0, batch.x.shape[0])
        mixup_x = lambda_ * x + (1-lambda_) * x[index]
        chiral = batch.chiral.clone().detach() 
        chiral_one_hot = torch.zeros(chiral.shape[0], 3, device=chiral.device)
        chiral_one_hot.scatter_(1, chiral.unsqueeze(1), 1)  
        mixup_chiral = lambda_ * chiral_one_hot + (1-lambda_) * chiral_one_hot[index] 
        batch.mixup_chiral = mixup_chiral  
        graph_indices = batch.batch
        mask = graph_indices.unsqueeze(0) == graph_indices.unsqueeze(1)
        mask = mask.to(batch.x.device)
        row, col = edge_index[0].clone(), edge_index[1].clone()
        row, col = index_new[row], index_new[col]
        edge_index_perm = torch.stack([row, col], dim=0)
        mixup_edge_attr = lambda_ * edge_attr + (1 - lambda_) * edge_attr[index_new[edge_index[0]]]
        edge_index_m, edge_attr_m = dropout_adj(edge_index, edge_attr=edge_attr, p=1 - lambda_, training=False)
        edge_index_perm, edge_attr_perm = dropout_adj(edge_index_perm, edge_attr=mixup_edge_attr, p=lambda_, training=False)
        edge_index_m = edge_index_m.to(batch.x.device)
        edge_index_perm = edge_index_perm.to(batch.x.device)
        mixup_edge_index = torch.cat((edge_index_m, edge_index_perm), dim=1)
        mixup_edge_attr = torch.cat((edge_attr_m, edge_attr_perm), dim=0)
        valid_edges = mask[mixup_edge_index[0], mixup_edge_index[1]]
        mixup_edge_index = mixup_edge_index[:, valid_edges]
        mixup_edge_attr = mixup_edge_attr[valid_edges]
        mixup_n_edges = calculate_edges_per_graph_undirected_modified(mixup_edge_index , batch).to(x.device)
        mixup_edge_batch = torch.repeat_interleave(graph_idx, mixup_n_edges, dim=0).to(x.device)
        mixup_x_final, mixup_edge_attr_final, mixup_u_final, mixup_chiral_list = self._forward_shared_then_specific(
            mixup_x, mixup_edge_attr, u, mixup_edge_index, mixup_edge_batch, node_batch, 
            num_nodes, mixup_n_edges, mode='inter'
        )
        return mixup_chiral_list[-1] if mixup_chiral_list else None
    def _compute_intra_mixup_enhanced(self, batch, x_embed, edge_attr_embed, u_embed,
                                    edge_index, edge_batch, node_batch, num_nodes, num_edges):
        line_graph_x = batch.line_graph_x
        line_graph_edge_index = batch.line_graph_edge_index  
        line_graph_edge_attr = batch.line_graph_edge_attr
        line_graph_chirality = batch.line_graph_chirality
        line_graph_n_nodes = batch.line_graph_n_nodes
        line_graph_n_edges = batch.line_graph_n_edges
        line_graph_x_embed = self.encoder_line_node(line_graph_x.to(x_embed.device))
        line_graph_edge_attr_embed = self.encoder_line_edge(line_graph_edge_attr.to(x_embed.device))
        line_graph_idx = torch.arange(batch.num_graphs).to(x_embed.device)
        line_graph_edge_batch = torch.repeat_interleave(line_graph_idx, line_graph_n_edges, dim=0)
        line_graph_node_batch = torch.repeat_interleave(line_graph_idx, line_graph_n_nodes, dim=0)
        line_x_final, line_edge_attr_final, line_u_final, line_graph_chiral_list = self._forward_shared_then_specific(
            line_graph_x_embed, line_graph_edge_attr_embed, u_embed, 
            line_graph_edge_index.to(x_embed.device), line_graph_edge_batch, line_graph_node_batch,
            line_graph_n_nodes, line_graph_n_edges, mode='intra'
        )
        return line_graph_chiral_list[-1] if line_graph_chiral_list else None
    def _compute_inter_mixup(self, batch, x_embed, edge_attr_embed, u_embed, 
                           edge_index, edge_batch, node_batch, num_nodes, num_edges):
        x, edge_attr, u = x_embed, edge_attr_embed, u_embed  
        lambda_ = self.mixup_ratio
        graph_idx = torch.arange(batch.num_graphs).to(x.device)
        index = torch.randperm(batch.x.shape[0])
        index_new = torch.zeros(batch.x.shape[0], dtype=torch.long)
        index_new[index] = torch.arange(0, batch.x.shape[0])
        mixup_x = lambda_ * x + (1-lambda_) * x[index]
        chiral = batch.chiral.clone().detach() 
        chiral_one_hot = torch.zeros(chiral.shape[0], 3, device=chiral.device)
        chiral_one_hot.scatter_(1, chiral.unsqueeze(1), 1)  
        mixup_chiral = lambda_ * chiral_one_hot + (1-lambda_) * chiral_one_hot[index]  
        batch.mixup_chiral = mixup_chiral  
        graph_indices = batch.batch
        mask = graph_indices.unsqueeze(0) == graph_indices.unsqueeze(1)
        mask = mask.to(batch.x.device)
        row, col = edge_index[0].clone(), edge_index[1].clone()
        row, col = index_new[row], index_new[col]
        edge_index_perm = torch.stack([row, col], dim=0)
        mixup_edge_attr = lambda_ * edge_attr + (1 - lambda_) * edge_attr[index_new[edge_index[0]]]
        edge_index_m, edge_attr_m = dropout_adj(edge_index, edge_attr=edge_attr, p=1 - lambda_, training=False)
        edge_index_perm, edge_attr_perm = dropout_adj(edge_index_perm, edge_attr=mixup_edge_attr, p=lambda_, training=False)
        edge_index_m = edge_index_m.to(batch.x.device)
        edge_index_perm = edge_index_perm.to(batch.x.device)
        mixup_edge_index = torch.cat((edge_index_m, edge_index_perm), dim=1)
        mixup_edge_attr = torch.cat((edge_attr_m, edge_attr_perm), dim=0)
        valid_edges = mask[mixup_edge_index[0], mixup_edge_index[1]]
        mixup_edge_index = mixup_edge_index[:, valid_edges]
        mixup_edge_attr = mixup_edge_attr[valid_edges]
        mixup_n_edges = calculate_edges_per_graph_undirected_modified(mixup_edge_index , batch).to(x.device)
        mixup_edge_batch = torch.repeat_interleave(graph_idx, mixup_n_edges, dim=0).to(x.device)
        mixup_cur_chiral = mixup_x.new_zeros((mixup_x.size(0), 3)).uniform_(-1, 1)
        extended_x, extended_edge_attr = self.extend_x_edge_mixup(mixup_cur_chiral, mixup_x, mixup_edge_attr, mixup_edge_index)
        mixup_x_final, mixup_edge_attr_final, mixup_u_final, mixup_chiral_list = self._forward_shared_then_specific(
            extended_x.to(x.device), mixup_edge_attr, u.to(x.device), 
            mixup_edge_index.to(x.device), mixup_edge_batch.to(x.device), node_batch.to(x.device),
            num_nodes.to(x.device), mixup_n_edges.to(x.device), mode='inter'
        )
        return mixup_chiral_list[-1] if mixup_chiral_list else None
    def _compute_intra_mixup(self, batch, x_embed, edge_attr_embed, u_embed,
                           edge_index, edge_batch, node_batch, num_nodes, num_edges):
        line_graph_x = batch.line_graph_x
        line_graph_edge_index = batch.line_graph_edge_index  
        line_graph_edge_attr = batch.line_graph_edge_attr
        line_graph_chirality = batch.line_graph_chirality
        line_graph_n_nodes = batch.line_graph_n_nodes
        line_graph_n_edges = batch.line_graph_n_edges
        line_graph_x_embed = self.encoder_line_node(line_graph_x.to(x_embed.device))
        line_graph_edge_attr_embed = self.encoder_line_edge(line_graph_edge_attr.to(x_embed.device))
        line_graph_idx = torch.arange(batch.num_graphs).to(x_embed.device)
        line_graph_edge_batch = torch.repeat_interleave(line_graph_idx, line_graph_n_edges, dim=0)
        line_graph_node_batch = torch.repeat_interleave(line_graph_idx, line_graph_n_nodes, dim=0)
        cur_line_chirality = line_graph_x_embed.new_zeros((line_graph_x_embed.size(0), 3)).uniform_(-1, 1)
        cur_line_chirality = line_graph_x_embed.new_zeros((line_graph_x_embed.size(0), 3)).uniform_(-1, 1)
        extended_line_x, extended_line_edge_attr = self.extend_x_edge_mixup(
            cur_line_chirality, line_x, line_edge_attr, line_graph_edge_index.to(x_embed.device)
        )
        line_x_final, line_edge_attr_final, line_u_final, line_graph_chiral_list = self._forward_shared_then_specific(
            extended_line_x, line_edge_attr, line_u,
            line_graph_edge_index.to(x_embed.device), line_graph_edge_batch, line_graph_node_batch,
            line_graph_n_nodes, line_graph_n_edges, mode='intra'
        )
        return line_graph_chiral_list[-1] if line_graph_chiral_list else None
    def compute_loss(self, chirality_list, extra_output, batch, args, 
                    inter_mixup_logits=None, intra_mixup_logits=None, mixup_soft_targets=None):
        loss_dict = {}
        total_loss = 0
        chirality = batch.chiral
        if "prior_chiral_list" in extra_output and extra_output["prior_chiral_list"] is not None:
            loss_tmp = F.cross_entropy(extra_output["prior_chiral_list"], chirality, ignore_index=0)
            total_loss = total_loss + loss_tmp
            loss_dict["loss_prior_chirality"] = loss_tmp
        if "latent_mean" in extra_output and "latent_logstd" in extra_output:
            mean = extra_output["latent_mean"]
            log_std = extra_output["latent_logstd"]
            kld = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std), dim=-1)
            kld = kld.mean()
            total_loss = total_loss + kld * args.vae_beta
            loss_dict["loss_kld"] = kld
        loss_tmp = F.cross_entropy(chirality_list[-1], chirality, ignore_index=0)
        total_loss = total_loss + loss_tmp
        loss_dict["loss_chirality_last"] = loss_tmp
        if inter_mixup_logits is not None and hasattr(self, 'inter_mixup_weight') and self.inter_mixup_weight > 0:
            if mixup_soft_targets is not None:
                mixed_targets = mixup_soft_targets
            else:
                perm = torch.randperm(chirality.size(0), device=chirality.device)
                lambda_ = self.mixup_ratio
                chirality_one_hot = F.one_hot(chirality, num_classes=3).float()
                mixed_targets = lambda_ * chirality_one_hot + (1 - lambda_) * chirality_one_hot[perm]
            inter_mixup_loss = self._soft_cross_entropy(inter_mixup_logits, mixed_targets)
            total_loss = total_loss + self.inter_mixup_weight * inter_mixup_loss
            loss_dict["loss_inter_mixup"] = inter_mixup_loss
        if intra_mixup_logits is not None and hasattr(self, 'intra_mixup_weight') and self.intra_mixup_weight > 0:
            mixed_targets = self._create_intra_mixup_targets(chirality, batch)
            intra_mixup_loss = self._soft_cross_entropy(intra_mixup_logits, mixed_targets)
            total_loss = total_loss + self.intra_mixup_weight * intra_mixup_loss
            loss_dict["loss_intra_mixup"] = intra_mixup_loss
        loss_dict["loss"] = total_loss
        return total_loss, loss_dict
    def _create_intra_mixup_targets(self, chirality, batch):
        line_graph_chirality = batch.line_graph_chirality.float()  
        mixed_targets = torch.zeros(line_graph_chirality.size(0), 3, device=line_graph_chirality.device)
        for i, chiral_val in enumerate(line_graph_chirality):
            if chiral_val <= 0.5: 
                mixed_targets[i, 0] = 1.0 - chiral_val
                if chiral_val > 0:
                    mixed_targets[i, 1] = chiral_val 
            elif chiral_val <= 1.5: 
                mixed_targets[i, 1] = 1.5 - chiral_val  
                mixed_targets[i, 2] = chiral_val - 0.5   
            else: 
                mixed_targets[i, 2] = chiral_val - 1.0
                if chiral_val < 2.0:
                    mixed_targets[i, 1] = 2.0 - chiral_val 
        row_sums = mixed_targets.sum(dim=1, keepdim=True)
        mixed_targets = mixed_targets / (row_sums + 1e-8)  
        lambda_ = self.mixup_ratio
        line_graph_n_nodes = batch.line_graph_n_nodes
        line_graph_idx = torch.arange(batch.num_graphs).to(mixed_targets.device)
        line_graph_node_batch = torch.repeat_interleave(line_graph_idx, line_graph_n_nodes, dim=0)
        for graph_id in range(batch.num_graphs):
            mask = (line_graph_node_batch == graph_id)
            graph_indices = torch.where(mask)[0]
            if len(graph_indices) > 1:
                perm = torch.randperm(len(graph_indices))
                original_targets = mixed_targets[graph_indices]
                permuted_targets = mixed_targets[graph_indices[perm]]
                mixed_targets[graph_indices] = lambda_ * original_targets + (1 - lambda_) * permuted_targets
        return mixed_targets
    def _soft_cross_entropy(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()
    def compute_loss_stage(self, chirality_list, extra_output, batch, args):
        loss_dict = {}
        total_loss = 0
        chirality = batch.chiral
        prior_loss = 1 * F.cross_entropy(extra_output["prior_chiral_list"], chirality)
        loss_dict["loss_prior_chirality"] = prior_loss
        if "latent_mean" in extra_output and "latent_logstd" in extra_output:
            mean = extra_output["latent_mean"]
            log_std = extra_output["latent_logstd"]
            kld = -0.5 * torch.mean(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std))
            total_loss += kld * args.vae_beta
            loss_dict["loss_kld"] = kld
        final_logits = chirality_list[-1]
        task1_labels = (chirality != 0).long() 
        task1_logits = torch.stack([
            final_logits[:, 0], 
            torch.logsumexp(final_logits[:, 1:3], dim=1)  
        ], dim=1)
        task1_loss = F.cross_entropy(task1_logits, task1_labels)
        mask = (chirality != 0)
        if mask.sum() > 0:
            task2_labels = chirality[mask] - 1
            task2_logits = final_logits[mask][:, 1:3] 
            task2_loss = F.cross_entropy(task2_logits, task2_labels)
        else:
            task2_loss = torch.tensor(0.0, device=final_logits.device)
        loss_chirality_last = 0.1 * task1_loss + 0.9 * task2_loss
        total_loss += loss_chirality_last
        loss_dict.update({
            "loss_task1": task1_loss,
            "loss_task2": task2_loss,
            "loss_chirality_last": loss_chirality_last,
            "loss": total_loss
        })
        return total_loss, loss_dict
    def compute_loss_oringin(self, chirality_list, extra_output, batch, args):
        loss_dict = {}
        chirality_labels = batch.chiral
        loss = F.cross_entropy(chirality_list[-1], chirality_labels, ignore_index=0)
        loss_dict["loss_chirality_last"] = loss
        loss_dict["loss"] = loss
        return loss, loss_dict
    def compute_loss_vae(self, chirality_list, extra_output, batch, args):
        loss_dict = {}
        loss = 0
        chirality = batch.chiral
        loss_tmp = F.cross_entropy(extra_output["prior_chiral_list"], 
                                chirality, ignore_index=0)
        loss = loss + loss_tmp
        loss_dict["loss_prior_chirality"] = loss_tmp
        if "latent_mean" in extra_output and "latent_logstd" in extra_output:
            mean = extra_output["latent_mean"]
            log_std = extra_output["latent_logstd"]
            kld = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std), dim=-1)
            kld = kld.mean()
            loss = loss + kld * args.vae_beta
            loss_dict["loss_kld"] = kld
        loss_tmp = F.cross_entropy(chirality_list[-1], 
                                chirality, ignore_index=0)
        loss = loss + loss_tmp
        loss_dict["loss_chirality_last"] = loss_tmp
        loss_dict["loss"] = loss
        return loss, loss_dict
    def compute_loss_origin_novae(self, chirality_list, extra_output, batch, args):
        loss_dict = {}
        chirality_labels = batch.chiral
        mask = (chirality_labels != 0) 
        if mask.sum() > 0: 
            filtered_logits = chirality_list[-1][mask] 
            filtered_labels = chirality_labels[mask]   
            loss = F.cross_entropy(filtered_logits, filtered_labels)
        else:
            loss = torch.tensor(0.0, device=chirality_labels.device)
        loss_dict["loss_chirality_last"] = loss
        loss_dict["loss"] = loss 
        return loss, loss_dict
    def extend_x_edge_mixup(self, chiral, x, edge_attr, edge_index):
        extended_x = x + self.chiral_embedding(chiral)
        row = edge_index[0]
        col = edge_index[1]
        sent_chiral = chiral[row]
        received_chiral = chiral[col]
        length = (sent_chiral - received_chiral).norm(dim=-1).unsqueeze(-1)
        extended_edge_attr = edge_attr + self.dis_embedding(length)
        return extended_x, extended_edge_attr
    def extend_x_edge2(self, chirality, x, edge_attr, edge_index):
        chirality = chirality.float() 
        extended_x = x + self.pos_embedding2(chirality)
        row = edge_index[0]
        col = edge_index[1]
        sent_chirality = chirality[row]
        received_chirality = chirality[col]
        length = (sent_chirality - received_chirality).norm(dim=-1).unsqueeze(-1)
        extended_edge_attr = edge_attr + self.dis_embedding(length)
        return extended_x, extended_edge_attr
    def random_augmentation(self, pos, batch):
        if self.rand_aug and self.training:
            return get_random_rotation_3d(pos)
        else:
            return pos
    def reparameterization(self, mean, log_std):
        std = torch.exp(log_std)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z
def one_hot_atoms_chiral(atoms):
    vocab_sizes = get_atom_feature_dims_chiral()
    one_hots = []
    for i in range(atoms.shape[1]):
        one_hots.append(
            F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)
def one_hot_bonds_chiral(bonds):
    vocab_sizes = get_bond_feature_dims()
    one_hots = []
    for i in range(bonds.shape[1]):
        one_hots.append(
            F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)