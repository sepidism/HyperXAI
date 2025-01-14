import torch
import utils


import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from layers import *
from infonce import *
from models import *

import copy
from tqdm import tqdm

import math 

from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch.optim import Adam
from torch_geometric.data import Data, Batch
import argparse
import random
import numpy as np
import os.path as osp




import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import *
from collections import defaultdict
from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_geometric.utils import dropout_adj, degree, to_undirected, k_hop_subgraph, add_self_loops,remove_isolated_nodes
from torch_geometric.data import Data

device = 1

class Attention(nn.Module):


    def __init__(self, embed_dim: int, hidden_dim: int = 16):

        super(Attention, self).__init__()

        self.embed_dims = embed_dim
        self.bias = Parameter(torch.Tensor(1)).to(device)

        self.emb_linear_node = nn.Sequential(nn.Linear(self.embed_dims, hidden_dim), nn.ReLU())
        self.emb_linear_hedge = nn.Sequential(nn.Linear(self.embed_dims, hidden_dim), nn.ReLU())


    

    
    def forward(self, n_x, e_x, is_source):

        w = 1
        if is_source:
            w = 1
        else:
            w = self.bias
        out_n = self.emb_linear_node(n_x) 
        out_e = w * self.emb_linear_hedge(e_x)
        out_n = torch.reshape(out_n, (-1, 16))
        out_e = torch.reshape(out_e, (-1, 16))

        out_e = torch.permute(out_e,(1,0))

        out = torch.matmul(out_n, out_e)
        return out


def parse_method(args, data):
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)


    return model

class AArgs():
    def __init__(self, dataset):
        self.runs = 20
        self.mode = "InfoNCE"
        self.train_prop = 0.1
        self.valid_prop = 0.25
        self.train_prop = 0.5
        self.lr = 0.001
        self.wd = 0.0
        self.epochs=500
        self.p_epoch=200
        self.aug_ratio=0.3
        self.t=0.3
        self.p_lr=0
        self.g_lr=1e-3
        self.step=1
        self.m_l=0
        self.hard = 1
        self.deg = 0
        self.sub_size = 16384
        self.seed= 123
        self.g_l= 1
        self.aug = "mask" 
        self.permute_self_edge = True
        self.add_e = True
        self.linear = False
        self.sub_size = 16384
        self.m_l = 0
        self.a_l = 0.1
        self.display_step = -1
        self.aggregate = 'mean'
        self.cuda = 1
        self.method = 'AllDeepSets'
        self.add_self_loop = True
        self.exclude_self = False
        self.normtype = 'all_one'
        self.UniGNN_degV = 0.0
        self.UniGNN_degE = 0.0
        self.normalization = 'ln'
        self.deepset_input_norm = True
        self.LearnMask = False
        self.PMA=True  
        self.GPR=False
        self.LearnMask=False
        self.HyperGCN_mediators=True
        self.HyperGCN_fast=True
        self.HCHA_symdegnorm=False
        self.dropout=0.5
        self.t= 0.5
        self.p_lr = 0
        self.p_epochs= 300
        self.aug_ratio = 0.1
        self.p_hidden = -1
        self.p_layer = -1
        if dataset == 'zoo':
            self.classes = 8
            self.All_num_layers = 1
            self.dname = dataset
            self.dropout = 0.2
            self.aggr = 'add'
            self.NormLayer = 'ln'
            self.InputNorm = True
            self.GPR = False
            self.LearnMask = False
            self.num_features = 0
            self.Classifier_hidden = 64
            self.num_classes = 0
            self.Classifier_num_layers = 1
            self.MLP_hidden = 64
            self.MLP_num_layers = 2
            self.p_layer = -1
            self.heads = 1
            self.PMA = True
            self.p_layer = -1
            self.p_hidden = -1
            self.feature_noise = 1
            self.method = 'AllDeepSets'
            self.add_self_loop = True
            self.exclude_self = True
            self.normtype = 'all_one'
            self.UniGNN_degV = 0.0
            self.UniGNN_degE = 0.0
            self.normalization = 'ln'
            self.deepset_input_norm = True
            self.g_l= 10
            self.a_l = 0.5

        elif dataset == 'cora':
            self.classes = 7
            self.All_num_layers = 1
            self.dname = 'cora'
            self.dropout = 0.2
            self.aggr = 'add'
            self.NormLayer = 'ln'
            self.InputNorm = True
            self.GPR = False
            self.LearnMask = False
            self.num_features = 0
            self.Classifier_hidden = 128
            self.num_classes = 0
            self.Classifier_num_layers = 1
            self.MLP_hidden = 256
            self.MLP_num_layers = 2
            self.p_layer = -1
            self.heads = 4
            self.PMA = True
            self.p_layer = -1
            self.p_hidden = -1
            self.feature_noise = 0.0
            self.method = 'AllDeepSets'
            self.add_self_loop = True
            self.exclude_self = True
            self.normtype = 'all_one'
            self.a_l = 0.5
            self.UniGNN_degV = 0.0
            self.UniGNN_degE = 0.0
            self.normalization = 'ln'
            self.deepset_input_norm = True
            self.g_l= 10
            
        elif dataset == 'citeseer':
            self.classes = 6
            self.All_num_layers = 1
            self.dname = 'citeseer'
            self.dropout = 0.2
            self.aggr = 'add'
            self.NormLayer = 'ln'
            self.InputNorm = True
            self.GPR = False
            self.LearnMask = False
            self.num_features = 0
            self.Classifier_hidden = 256
            self.num_classes = 0
            self.Classifier_num_layers = 1
            self.MLP_hidden = 512
            self.MLP_num_layers = 2
            self.p_layer = -1
            self.heads = 8
            self.PMA = True
            self.p_layer = -1
            self.p_hidden = -1
            self.a_l = 0.5
            self.feature_noise = 0.0

            
        elif dataset == 'pubmed':
            self.classes = 3
            self.wd = 0.0
            self.g_l = 10
            self.a_l = 0.5
            self.All_num_layers = 1
            self.dname = 'pubmed'
            self.Classifier_hidden = 256
            self.Classifier_num_layers = 1
            self.MLP_hidden = 256
            self.MLP_num_layers = 2
            self.heads = 8
            self.feature_noise = 0.0
            self.method = 'AllDeepSets'
            
 
        elif dataset == 'coauthor_cora':
            self.classes = 7
            self.All_num_layers = 1
            self.dname = 'coauthor_cora'
            self.dropout = 0.2
            self.aggr = 'add'
            self.NormLayer = 'ln'
            self.InputNorm = True
            self.GPR = False
            self.LearnMask = False
            self.num_features = 0
            self.Classifier_hidden = 128
            self.num_classes = 0
            self.Classifier_num_layers = 1
            self.MLP_hidden = 128
            self.MLP_num_layers = 2
            self.p_layer = -1
            self.heads = 8
            self.PMA = True
            self.p_layer = -1
            self.p_hidden = -1
            self.feature_noise = 0.0
            self.method = 'AllDeepSets'
            self.add_self_loop = True
            self.exclude_self = True
            self.normtype = 'all_one'
            self.a_l = 0.3
            self.UniGNN_degV = 0.0
            self.UniGNN_degE = 0.0
            self.normalization = 'ln'
            self.deepset_input_norm = True
            self.g_l= 10

class HyperExplainer(nn.Module):
    
    def __init__(self, model, embed_dim: int, device = 1, explain_graph: bool = True, 
        coff_size: float = 0.01, coff_ent: float = 5e-4, grad_scale: float = 0.25,
        loss_type = 'NCE', t0: float = 5.0, t1: float = 1.0, top_k = 5, classes = 3, num_hops: Optional[int] = None):

        super(HyperExplainer, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.explain_graph = explain_graph
        self.model = model.to(self.device)

        self.explainer = Attention(embed_dim).to(device)

        self.grad_scale = grad_scale
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.loss_type = loss_type
        self.sub_size = top_k
        self.sub_hop = 3
        
        self.classes = classes

        self.S = None




    def __loss__(self, embed: Tensor, pruned_embed: Tensor,**kwargs):

        max_items = kwargs.get('max_items')

        contrast_loss = NCE_loss([embed, pruned_embed])

        return contrast_loss

    
    

    def star_subgraph(self, data, node_id, embed): 


        data_tmp = copy.deepcopy(data)
        clique_data = ConstructV2V(data_tmp)
        sub_size = self.sub_size
       

        subset, _, _, _ = k_hop_subgraph(node_id, self.sub_hop, clique_data.edge_index)
        

        mask_hedges = torch.zeros(data.edge_index.shape[1],dtype=torch.bool).to(device)
        subset.to(device)
        data.to(device)
        for node in subset:
            mask_hedges |= (data.edge_index[0] == node)



        subset_dest_nodes = data.edge_index[1,mask_hedges]

        

        combined_tensor = torch.cat((subset_dest_nodes.to(device), subset.to(device)), dim=0)


        
        node_mask = torch.zeros(data.edge_index.max().item() + 1, dtype=torch.bool).to(device)
        node_mask[combined_tensor] = True

        rows, cols = data.edge_index
        node_mask = (node_mask[rows] & node_mask[cols]).squeeze()


        old_edge_index  = data.edge_index[:, node_mask] # all the edges  
        selected_nodes, selected_hedges = old_edge_index
        unique_nodes = torch.unique(selected_nodes.flatten())
        unique_hedges = torch.unique(selected_hedges.flatten())
        
        
        new_node_index = torch.arange(unique_nodes.size(0), dtype=torch.long)
        new_hedge_index = torch.arange(unique_hedges.size(0), dtype=torch.long)
        mapping = {unique_nodes[i].item(): new_node_index[i].item() for i in range(unique_nodes.size(0))}
        mapping_h = {unique_hedges[i].item(): (new_hedge_index[i].item() + unique_nodes.size(0)) for i in range(unique_hedges.size(0))}
        mapping.update(mapping_h)
        new_edge_index = torch.tensor([mapping[e.item()] for e in old_edge_index.flatten()], dtype=torch.long).reshape(old_edge_index.shape)
        n_xx = data.x[unique_nodes]

        new_data = Data(
            x=data.x[unique_nodes],  
            edge_index=new_edge_index,
            edge_attr=data.edge_attr,
            y = data.y[unique_nodes],
            n_x = torch.tensor([n_xx.shape[0]]), 
            train_percent=data.train_percent,
            num_hyperedges=new_hedge_index.size(),
            totedges=new_hedge_index.size(),
            num_ori_edge=new_edge_index.shape[1],
            norm=torch.ones_like(new_edge_index[0]))
        

        if old_edge_index.shape[1] < sub_size:
            sub_size = old_edge_index.shape[1] - 1
            self.sub_size = sub_size

        

        new_emb = embed[unique_nodes]
        weights = defaultdict(list)
        edge_weights = []
        for i in range(new_edge_index.shape[1]): 
            n, e = new_edge_index[:,i]
            node_emb = new_emb[n]
            edges = new_edge_index[:,new_edge_index[1] == e]
            neighbor_nodes = set(edges[0].tolist())
            ne_emb = embed[list(neighbor_nodes)]
            e_emb = torch.mean(new_emb[list(neighbor_nodes)], dim=0)
            if n == torch.tensor(mapping[node_id]):
                is_source = True
            else:
                is_source = False
            w_ne = self.explainer(node_emb.to(device), e_emb.to(device), is_source)

            edge_weights.append(w_ne.flatten())

        softmax_weights = []
        for i in range(new_edge_index.shape[1]):
            src_node = new_edge_index[0, i]
            weights_n = edge_weights[i]
            softmax_n = F.softmax(weights_n, dim=0)
            softmax_weights.append(weights_n)
        
        new_norm = torch.tensor(edge_weights)
        
        sub_size = min(sub_size, new_norm.size(0)-1)
        
        topk_indices = torch.topk(new_norm, k=sub_size, largest=True).indices
        all_imp_nodes = new_edge_index[:,topk_indices]
        imp_nodes, imp_hedges = all_imp_nodes
        unique_imp_nodes = torch.unique(imp_nodes.flatten())
        imp_nodes_size = len(unique_imp_nodes.tolist())
        

        not_imp = []
        
        for i in range (0,new_norm.size(0)):
            if i not in topk_indices.tolist():
                not_imp.append(i)
        not_topk = torch.LongTensor(not_imp)
   
        
        topk_edge_index = new_edge_index[:,not_topk]
        topk_edge_weight = new_norm[not_topk]
        
        
        is_selected = True
        selected_nodess, selected_hedgess = topk_edge_index
        if mapping[node_id] not in selected_nodess:
            is_selected = False
            

        unique_nodess = torch.unique(selected_nodess.flatten())
        unique_hedgess = torch.unique(selected_hedgess.flatten())
        
        
        new_node_indexs = torch.arange(unique_nodess.size(0), dtype=torch.long)
        new_hedge_indexs = torch.arange(unique_hedgess.size(0), dtype=torch.long)
        mappings = {unique_nodess[i].item(): new_node_indexs[i].item() for i in range(unique_nodess.size(0))}
        mapping_hs = {unique_hedgess[i].item(): (new_hedge_indexs[i].item() + unique_nodess.size(0)) for i in range(unique_hedgess.size(0))}
        mappings.update(mapping_hs)
        new_edge_indexs = torch.tensor([mappings[e.item()] for e in topk_edge_index.flatten()], dtype=torch.long).reshape(topk_edge_index.shape)

        n_xx = new_data.x[unique_nodess]
        
        new_datas = Data(
            x=new_data.x[unique_nodess],  
            edge_index=new_edge_indexs,
            edge_attr=new_data.edge_attr,
            y = new_data.y[unique_nodess],
            n_x = torch.tensor([n_xx.shape[0]]), 
            train_percent=data.train_percent,
            num_hyperedges=new_hedge_indexs.size(),
            totedges=new_hedge_indexs.size(),
            num_ori_edge=new_edge_indexs.shape[1],
            norm=topk_edge_weight) #torch.ones_like(new_edge_index[0]),

        if is_selected:
            return new_datas, float(imp_nodes_size/subset.size(0)), mappings[mapping[node_id]], is_selected
        else:
            return new_datas, float(imp_nodes_size/subset.size(0)), node_id, is_selected
    
    
    def train_explainer_node(self, data, train_list, batch_size=4, lr=0.001, epochs=10):

        
        optimizer = Adam(self.explainer.parameters(), lr=lr)
        for epoch in range(epochs):
            self.model.eval()
            self.explainer.train()

            
            loss = 0.0
            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
            with torch.no_grad():
                self.model.cpu()
                data.cpu()
                data_tmp = copy.deepcopy(data)
                
                out = self.model(data_tmp) # get the embedding
                all_embeds = F.softmax(out, dim=1)

            self.model.to(self.device)
            data.to(self.device)
            all_embeds.to(device)

            new_all_embeds = []           
            pruned_embeds = []
            masks = []
            for node_idx in (train_list):

                subgraph, imp_nodes_size, new_node_idx, is_selected = self.star_subgraph(data, node_idx, all_embeds)
                if new_node_idx == -1:
                    continue

                subgraph.to(device)
                out_t = self.model(subgraph)
                pruned_embed = F.softmax(out_t, dim=1)
                
                if is_selected:
                    t_emb = pruned_embed[new_node_idx]
                    pruned_embeds.append(t_emb)

                    new_embed = all_embeds[node_idx]
                    new_all_embeds.append(new_embed)


                else:
                    continue
                    t_emb = torch.zeros(self.classes, dtype=float).to(device)
                    pruned_embeds.append(t_emb)
                

            new_all_embeds = torch.cat(new_all_embeds, 0)
            new_all_embeds = torch.reshape(new_all_embeds, (-1,self.classes))
            pruned_embeds = torch.cat(pruned_embeds, 0).to(device)
            pruned_embeds = torch.reshape(pruned_embeds, (-1,self.classes))

            loss = self.__loss__(new_all_embeds.to(device), pruned_embeds.to(device))

            loss.backward()
            optimizer.step()


    def __edge_to_node__(self, subgraph, old_edge_mask):
        

        nodes, edges = subgraph.edge_index
        unique_nodes = torch.unique(nodes.flatten())
        unique_edges = torch.unique(edges.flatten())
        subgraph_node_mask = torch.cat((unique_nodes, unique_edges))
                
        all_nodes, all_edges = old_edge_mask
        unique_all_nodes = torch.unique(all_nodes.flatten())
        unique_all_edges = torch.unique(all_edges.flatten())
        all_node_mask = torch.cat((unique_all_nodes, unique_all_edges))
        return all_node_mask, float(subgraph_node_mask.size(0) / all_node_mask.size(0))



    def forward(self, data, **kwargs):
       
        self.sub_size = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
        node_idx = kwargs.get('node_idx')
        self.model.eval()


        data = data.to(self.device)
        self.model.to(device)
        if node_idx is not None:
            t_data = copy.deepcopy(data)
            node_embed = self.model(t_data)
            node_embed = F.softmax(node_embed, dim=1)
            embed = node_embed[node_idx]
        else:
            assert node_idx is not None, "please input the node_idx"

        subgraph, sparsity, new_node_idx, is_selected = self.star_subgraph(data, node_idx, node_embed) 
        if new_node_idx == -1:
            related_preds = [{
            'fidelity': -1,
            'sparsity': -1}]
            return related_preds 

        new_embed = self.model(subgraph.to(device))
        new_embed = F.softmax(new_embed, dim=1)
        if is_selected:
            masked_embed = new_embed[new_node_idx]
        else:
            index = embed.argmax()
            masked_embed = torch.zeros(self.classes, dtype=float)
            
        if masked_embed.argmax() == embed.argmax():
            fidelity = 0
        else:
            fidelity = 1

       
        related_preds = [{
            'fidelity': fidelity,
            'sparsity': sparsity}]
        return related_preds


    def mask_fn(data, node_mask: np.array):
        row, col = data.edge_index
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        ret_edge_index = data.edge_index[:, edge_mask]
        ret_edge_attr = None if data.edge_attr is None else data.edge_attr[edge_mask] 
        data = Data(x=data.x, edge_index=ret_edge_index, 
            edge_attr=ret_edge_attr, batch=data.batch)
        return data