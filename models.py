import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
# Import pooling and distance functions from PyG.  Only global_add_pool and
# radius are needed since RNA- and protein-specific functionality has been
# removed.
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops

from layers import Global_MessagePassing, Local_MessagePassing, Local_MessagePassing_s, \
    BesselBasisLayer, SphericalBasisLayer, MLP

class Config(object):
    """Simple configuration object for PAMNet.

    Only the QM9 dataset is supported in this simplified version.  The
    ``dataset`` attribute is still accepted for backwards compatibility,
    but its value is ignored.  The remaining attributes control the size
    of the network and cutoff radii used in the local and global layers.
    """

    def __init__(self, dataset: str, dim: int, n_layer: int,
                 cutoff_l: float, cutoff_g: float,
                 flow: str = 'source_to_target') -> None:
        # store the dataset name for reference but do not act on it
        self.dataset = dataset
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff_l = cutoff_l
        self.cutoff_g = cutoff_g
        self.flow = flow

class PAMNet(nn.Module):
    #''' 
    # ORIGINAL ONE2
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5):
    '''
    
    # MINE2
    def __init__(self, config: Config, num_spherical=3, num_radial=4, envelope_exponent=5):
    '''
        
        super(PAMNet, self).__init__()

        # store configuration
        # In this simplified implementation only QM9 is supported.  The
        # ``dataset`` attribute of the configuration is retained for
        # completeness, but no conditional logic depends on it.
        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g

        # Embedding table for atom types in QM9.  QM9 contains up to
        # five heavy atom species (H, C, N, O, F).  We omit the PDBbind
        # initialization network and RNA special cases from the original
        # repository.
        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        #'''  
        # ORIGINAL CODE2
        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([16, self.dim])
        self.mlp_rbf_l = MLP([16, self.dim])    
        self.mlp_sbf1 = MLP([num_spherical * num_radial, self.dim])
        self.mlp_sbf2 = MLP([num_spherical * num_radial, self.dim])
        '''

        #MINE2
        self.rbf_g = BesselBasisLayer(8, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(8, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([8, self.dim])
        self.mlp_rbf_l = MLP([8, self.dim])    
        self.mlp_sbf1 = MLP([num_spherical * num_radial, self.dim])  # 3*4=12
        self.mlp_sbf2 = MLP([num_spherical * num_radial, self.dim])


        #'''
        # ORIGINAL CODE1
        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing(config))
        '''
        # MINE1
        # نسخه‌ی weight-sharing: فقط یک لایه global و یک لایه local
        # که چند بار پشت سر هم روی x اعمال می‌شوند.
        self.global_layer = Global_MessagePassing(config)
        self.local_layer = Local_MessagePassing(config)
        '''

        self.softmax = nn.Softmax(dim=-1)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def get_edge_info(self, edge_index, pos):
        edge_index, _ = remove_self_loops(edge_index)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        """Compute model outputs for a batch of QM9 graphs.

        This implementation assumes that the input ``data`` object
        contains the attributes ``x`` (atom type indices), ``pos`` (3D
        coordinates), ``edge_index`` (bond connectivity) and ``batch``.
        All graphs in the batch are treated as small molecules from the QM9
        dataset.  Any dataset‑specific logic from the original
        implementation has been removed.
        """
        x_raw = data.x
        batch = data.batch
        pos = data.pos
        edge_index_l = data.edge_index

        # Lookup initial node embeddings based on atom types.
        x = torch.index_select(self.embeddings, 0, x_raw.long())

        # Build global adjacency: connect all pairs of atoms within cutoff_g
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, dist_g = self.get_edge_info(edge_index_g, pos)

        # Compute distances for the local (bond) graph.  Remove self‑loops
        # and compute distances between connected atoms.
        edge_index_l, dist_l = self.get_edge_info(edge_index_l, pos)

        # Precompute various index tensors required by the local message passing
        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = (
            self.indices(edge_index_l, num_nodes=x.size(0))
        )

        # Compute two‑hop angles in the local graph
        pos_ji = pos[idx_j] - pos[idx_i]
        pos_kj = pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj, dim=-1).norm(dim=-1)
        angle2 = torch.atan2(b, a)

        # Compute one‑hop angles in the local graph
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair = pos_j1_pair - pos_i_pair
        pos_jj_pair = pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair, dim=-1).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        # Basis function embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf1 = self.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = self.sbf(dist_l, angle2, idx_kj)

        # Project basis functions to hidden dimension
        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = self.mlp_sbf1(sbf1)
        edge_attr_sbf2 = self.mlp_sbf2(sbf2)

        # Message passing
        out_global: list[torch.Tensor] = []
        out_local: list[torch.Tensor] = []
        att_score_global: list[torch.Tensor] = []
        att_score_local: list[torch.Tensor] = []

        #'''
        # ORIGINAL ONE1
        for layer in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            x, out_l, att_score_l = self.local_layer[layer](
                x,
                edge_attr_rbf_l,
                edge_attr_sbf2,
                edge_attr_sbf1,
                idx_kj,
                idx_ji,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            out_local.append(out_l)
            att_score_local.append(att_score_l)
        '''

        # MINE1
        # weight-sharing: همان لایه‌ی global/local را n_layer بار تکرار می‌کنیم
        for _ in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer(
                x,
                edge_attr_rbf_g,
                edge_index_g,
            )
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            x, out_l, att_score_l = self.local_layer(
                x,
                edge_attr_rbf_l,
                edge_attr_sbf2,
                edge_attr_sbf1,
                idx_kj,
                idx_ji,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            out_local.append(out_l)
            att_score_local.append(att_score_l)
            '''


        
        # Attention‑based fusion of local and global representations
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), dim=-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weight = self.softmax(att_score)

        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), dim=-1)
        out = (out * att_weight).sum(dim=-1)
        out = out.sum(dim=0).unsqueeze(-1)

        # Aggregate node‑level outputs to graph‑level predictions
        out = global_add_pool(out, batch)
        return out.view(-1)


class PAMNet_s(nn.Module):

    #''' 
    # ORIGINAL ONE2
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5):
    '''

    #MINE2
    def __init__(self, config: Config, num_spherical=3, num_radial=4, envelope_exponent=5):
    '''
        
        super(PAMNet_s, self).__init__()

        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        #''' 
        # ORIGINAL ONE2
        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([16, self.dim])
        self.mlp_rbf_l = MLP([16, self.dim])    
        self.mlp_sbf = MLP([num_spherical * num_radial, self.dim])
        '''
        
        #MINE2
        self.rbf_g = BesselBasisLayer(8, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(8, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([8, self.dim])
        self.mlp_rbf_l = MLP([8, self.dim])    
        self.mlp_sbf = MLP([num_spherical * num_radial, self.dim])  # 3*4=12

        '''
        # ORIGINAL ONE1
        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing_s(config))
        '''

        # MINE1
        # weight-sharing در نسخه‌ی ساده‌تر مدل
        self.global_layer = Global_MessagePassing(config)
        self.local_layer = Local_MessagePassing_s(config)
        '''

        self.softmax = nn.Softmax(dim=-1)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        """Forward pass for the simplified PAMNet_s model on QM9 graphs.

        This version omits the dataset check present in the original
        implementation since only QM9 is supported.
        """
        x_raw = data.x
        edge_index_l = data.edge_index
        pos = data.pos
        batch = data.batch
        x = torch.index_select(self.embeddings, 0, x_raw.long())

        # Compute pairwise distances in the local layer
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Compute pairwise distances in the global layer
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(edge_index_l, num_nodes=x.size(0))

        # Compute one-hop angles in the local layer
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair = pos_j1_pair - pos_i_pair
        pos_jj_pair_vec = pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair_vec).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair_vec, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        # Get radial and spherical basis embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf = self.sbf(dist_l, angle, idx_jj_pair)

        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf = self.mlp_sbf(sbf)

        # Message passing
        out_global: list[torch.Tensor] = []
        out_local: list[torch.Tensor] = []
        att_score_global: list[torch.Tensor] = []
        att_score_local: list[torch.Tensor] = []

        #'''
        # ORIGINAL ONE1
        for layer in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            x, out_l, att_score_l = self.local_layer[layer](
                x,
                edge_attr_rbf_l,
                edge_attr_sbf,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            out_local.append(out_l)
            att_score_local.append(att_score_l)
        '''

        # MINE1
        # weight-sharing در نسخه‌ی ساده
        for _ in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer(
                x,
                edge_attr_rbf_g,
                edge_index_g,
            )
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            x, out_l, att_score_l = self.local_layer(
                x,
                edge_attr_rbf_l,
                edge_attr_sbf,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            out_local.append(out_l)
            att_score_local.append(att_score_l)
            '''

        
        # Fusion of local and global representations
        att_score = torch.cat((torch.cat(att_score_global, dim=0), torch.cat(att_score_local, dim=0)), dim=-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weight = self.softmax(att_score)

        out = torch.cat((torch.cat(out_global, dim=0), torch.cat(out_local, dim=0)), dim=-1)
        out = (out * att_weight).sum(dim=-1)
        out = out.sum(dim=0).unsqueeze(-1)
        out = global_add_pool(out, batch)
        return out.view(-1)
