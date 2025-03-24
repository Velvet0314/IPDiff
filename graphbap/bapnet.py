"""
BAPNet是一个等变图神经网络，专门用于处理蛋白质-配体相互作用。其网络构建流程如下：

1. 初始化特征表示层：
    - 为配体原子类型、蛋白质原子类型和氨基酸残基类型创建嵌入层
    - 配备融合层，合并蛋白质的原子和残基特征
    - 创建身份嵌入区分配体和蛋白质节点
2. 构建三个并行子网络：
    - 复合物图网络：处理整个蛋白质-配体复合物
    - 配体图网络：单独处理配体结构
    - 口袋图网络：单独处理蛋白质结合口袋
    - 每个子网络由多个等变块组成，每个等变块包含多层图卷积和坐标更新层
3.构建融合网络：
    - 使用图注意力网络(GAT)融合三个子网络的特征
    - 将复合物特征与合并后的配体-口袋特征进行融合
4.特征提取流程：
    - 首先进行坐标中心化处理
    - 创建各个分子实体的图结构(边)
    - 计算边的几何特征(距离)
    - 通过三个子网络并行传播信息
    - 融合子网络的信息，得到最终特征表示
5.关键创新点：
    - 等变性处理：保持3D分子结构的旋转和平移等变性
    - 多视角架构：同时考虑整体复合物和单独组件的视角
    - 预训练与迁移：加载预训练权重并冻结，作为特征提取器使用
这种架构设计使BAPNet能够捕获蛋白质-配体相互作用的复杂几何和化学特征，为下游任务提供有意义的分子表示。在IPDiff框架中，它作为条件网络引导分子生成过程。
"""

import os
from torch import nn
import torch
import math
from torch_geometric.nn import GATConv
from torch_geometric.nn import TopKPooling
from torch_scatter import scatter_add, scatter_mean

# 定义配体原子的基本类型列表
ligand_atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']

# 定义配体原子类型加上芳香性信息(数字1、2表示不同的芳香性状态)
ligand_atom_add_aromatic_types = ['H', 'C1', 'C2', 'N1', 'N2', 'O1', 'O2', 'F', 'P1', 'P2', 'S1', 'S2', 'Cl']

# 定义蛋白质结合口袋中常见的原子类型
pocket_atom_types = ['H', 'C', 'N', 'O', 'S', 'Se']

# 20种标准氨基酸的三字母缩写列表
residue_types = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

def get_edges(mask, x=None, edge_cutoff=None):
    """
    基于掩码和可选的距离阈值生成图的边
    
    参数:
        mask: 节点掩码，用于标识相同分子/结构中的原子
        x: 节点坐标，当使用距离阈值时需要提供
        edge_cutoff: 距离阈值，超过此距离的节点对将不形成边
        
    返回:
        edges: 形状为[2, num_edges]的边索引张量
    """
    adj = mask[:, None] == mask[None, :]  # 创建邻接矩阵，标识相同掩码值的节点对
    if edge_cutoff is not None:
        adj = adj & (torch.cdist(x, x) <= float(edge_cutoff))  # 添加距离约束
    edges = torch.stack(torch.where(adj), dim=0)  # 将邻接矩阵转换为边列表
    return edges

def remove_mean_batch_ligand(x_lig, x_pocket, lig_indices, pocket_indices):
    """
    分别减去配体和蛋白质口袋的质心，使两者都以各自的质心为中心
    
    参数:
        x_lig: 配体原子的坐标
        x_pocket: 口袋原子的坐标
        lig_indices: 配体批次索引
        pocket_indices: 口袋批次索引
        
    返回:
        中心化后的配体和口袋坐标
    """
    lig_mean = scatter_mean(x_lig, lig_indices, dim=0)  # 计算每个批次中配体的平均位置
    pocket_mean = scatter_mean(x_pocket, pocket_indices, dim=0)  # 计算每个批次中口袋的平均位置

    x_lig = x_lig - lig_mean[lig_indices]  # 减去配体的平均位置
    x_pocket = x_pocket - pocket_mean[pocket_indices]  # 减去口袋的平均位置
    return x_lig, x_pocket

def remove_lig_mean_batch_ligand(x_lig, x_pocket, lig_indices, pocket_indices):
    """
    减去配体的质心，使整个系统以配体的质心为中心
    
    参数:
        x_lig: 配体原子的坐标
        x_pocket: 口袋原子的坐标
        lig_indices: 配体批次索引
        pocket_indices: 口袋批次索引
        
    返回:
        以配体质心为中心的配体和口袋坐标
    """
    lig_mean = scatter_mean(x_lig, lig_indices, dim=0)  # 计算每个批次中配体的平均位置

    x_lig = x_lig - lig_mean[lig_indices]  # 减去配体的平均位置
    x_pocket = x_pocket - lig_mean[pocket_indices]  # 同样用配体平均位置处理口袋坐标
    return x_lig, x_pocket

def remove_pocket_mean_batch_ligand(x_lig, x_pocket, lig_indices, pocket_indices):
    """
    减去口袋的质心，使整个系统以口袋的质心为中心
    
    参数:
        x_lig: 配体原子的坐标
        x_pocket: 口袋原子的坐标
        lig_indices: 配体批次索引
        pocket_indices: 口袋批次索引
        
    返回:
        以口袋质心为中心的配体和口袋坐标
    """
    pocket_mean = scatter_mean(x_pocket, pocket_indices, dim=0)  # 计算每个批次中口袋的平均位置

    x_lig = x_lig - pocket_mean[lig_indices]  # 用口袋平均位置处理配体坐标
    x_pocket = x_pocket - pocket_mean[pocket_indices]  # 减去口袋的平均位置
    return x_lig, x_pocket

class BAPNet(nn.Module):
    """
    BAP (Binding Affinity Prediction) 网络，用于预测蛋白质-配体结合构象和亲和力
    
    该网络使用等变图神经网络处理3D分子结构，包含三个主要子网络:
    1. 复合物图网络: 处理蛋白质-配体复合物整体
    2. 配体图网络: 单独处理配体结构
    3. 口袋图网络: 单独处理蛋白质结合口袋
    
    最终通过融合网络集成这些特征进行预测
    """
    
    def __init__(self, ckpt_path=None,
                 hidden_nf: int = 128,
                 act_fn=nn.SiLU(), GAT_head: int = 2, graph_layers: int = 1, 
                 attention=False,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=1,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 edge_cutoff=None, ignore_keys: list = []):
        """
        初始化BAPNet网络
        
        参数:
            ckpt_path: 预训练权重加载路径
            hidden_nf: 隐藏层特征维度
            act_fn: 激活函数
            GAT_head: 图注意力网络头数量
            graph_layers: 图卷积层数量
            attention: 是否使用注意力机制
            norm_diff: 是否归一化坐标差异
            tanh: 是否使用tanh激活
            coords_range: 坐标范围限制
            norm_constant: 坐标归一化常数
            inv_sublayers: 每个等变块中的子层数
            sin_embedding: 是否使用正弦嵌入表示距离
            normalization_factor: 聚合操作的归一化因子
            aggregation_method: 特征聚合方法 ('sum'或'mean')
            edge_cutoff: 边剪裁距离阈值
            ignore_keys: 加载检查点时忽略的键列表
        """
        super(BAPNet, self).__init__()
        
        graph_dim = hidden_nf
        self.graph_dim = graph_dim
        self.hidden_nf = hidden_nf
        self.graph_layers = graph_layers
        
        # 嵌入层定义 - 将原子和残基类型转换为特征向量
        self.ligand_atom_type_embed = nn.Embedding(len(ligand_atom_add_aromatic_types) + 1, graph_dim)
        self.pocket_atom_type_embed = nn.Embedding(len(pocket_atom_types) + 1, graph_dim)
        self.pocket_residue_type_embed = nn.Embedding(len(residue_types) + 1, graph_dim)
        self.pocket_type_fusion = nn.Linear(graph_dim * 2, graph_dim)  # 融合口袋原子和残基特征
        
        # ID嵌入用于区分配体和蛋白质
        self.id_embed = nn.Embedding(2, 4)  # 0表示配体，1表示蛋白质
        self.embed_fusion = nn.Linear(graph_dim + 4, graph_dim)  # 融合类型和ID嵌入

        self.edge_cutoff = edge_cutoff

        # 坐标处理相关参数
        self.coords_range_layer = float(coords_range/1)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        
        # 距离嵌入
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2  # 使用正弦嵌入表示距离
        else:
            self.sin_embedding = None
            edge_feat_nf = 2  # 简单距离表示
            
        # 构建三个主要网络组件
        
        # 1. 复合物图网络 - 处理整个复合物
        self.ComplexesGraph = nn.ModuleList([
            EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method)
        ])

        # 2. 配体图网络 - 单独处理配体
        self.LigandGraph = nn.ModuleList([
            EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method)
        ])
        
        # 3. 口袋图网络 - 单独处理蛋白质口袋
        self.PocketGraph = nn.ModuleList([
            EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method)
        ])        

        # 添加更多层（如果graph_layers > 1）
        for layer_i in range(graph_layers - 1):
            # 为三个主要网络添加额外的等变块
            self.ComplexesGraph.append(
                EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method))

            self.LigandGraph.append(
                EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method))

            self.PocketGraph.append(
                EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method))

        # 融合网络 - 集成来自不同子网络的特征
        self.FusionGraph = nn.ModuleList([])
        self.FusionGraph.append(GATConv(graph_dim * 2, graph_dim * 1, GAT_head, concat=False))

        # 输出层和最终预测
        self.OutputLayer = nn.Sequential(nn.Linear(graph_dim * 1, graph_dim), nn.Hardswish(), nn.Linear(graph_dim, graph_dim))
        self.FinalOutput = nn.Linear(graph_dim * 1, 1)

        # 加载预训练权重
        assert ckpt_path is not None, "ckpt_path is None"
        assert os.path.exists(ckpt_path), "ckpt_path is not exist"
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.freeze_the_model()  # 加载后冻结模型参数

    def freeze_the_model(self):
        """
        冻结模型所有参数，使其不参与梯度更新
        """
        self.eval()  # 设置为评估模式
        for param in self.parameters():
            param.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        """
        从检查点加载模型权重
        
        参数:
            path: 检查点文件路径
            ignore_keys: 加载时忽略的键列表
        """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def extract_features(self, lig_coords, pocket_coords, lig_a_hidx, pocket_a_hidx, pocket_r_hidx, lig_mask, pocket_mask):
        """
        提取蛋白质-配体复合物的特征
        
        参数:
            lig_coords: 配体原子坐标
            pocket_coords: 口袋原子坐标
            lig_a_hidx: 配体原子类型索引
            pocket_a_hidx: 口袋原子类型索引
            pocket_r_hidx: 口袋残基类型索引
            lig_mask: 配体批次掩码
            pocket_mask: 口袋批次掩码
            
        返回:
            配体和口袋的特征表示
        """
        # 以口袋质心为中心对齐坐标
        lig_coords, pocket_coords = remove_pocket_mean_batch_ligand(lig_coords, pocket_coords, lig_mask, pocket_mask)
        
        device = lig_coords.device
        num_lig = lig_coords.shape[0]

        # 合并配体和口袋信息
        complexes_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        complexes_coords = torch.cat([lig_coords, pocket_coords], dim=0).to(torch.float32)
        complexes_id = torch.LongTensor([0] * lig_coords.shape[0] + [1] * pocket_coords.shape[0]).to(device)

        # 获取类型索引
        lig_atom_type = lig_a_hidx
        pocket_atom_type = pocket_a_hidx
        pocket_residue_type = pocket_r_hidx

        # 获取嵌入表示
        complexes_id_emb = self.id_embed(complexes_id)  # 配体/口袋的索引ID嵌入
        lig_atom_type_emb = self.ligand_atom_type_embed(lig_atom_type)  # 配体原子类型嵌入
        pocket_atom_type_emb = self.pocket_atom_type_embed(pocket_atom_type)  # 口袋原子类型嵌入
        pocket_residue_type_emb = self.pocket_residue_type_embed(pocket_residue_type)  # 口袋残基类型嵌入
        
        # 融合口袋的原子和残基特征
        pocket_type_emb = torch.cat([pocket_atom_type_emb, pocket_residue_type_emb], dim=1)
        pocket_type_emb = self.pocket_type_fusion(pocket_type_emb)

        # 合并配体和口袋特征
        complexes_type_emb = torch.cat([lig_atom_type_emb, pocket_type_emb], dim=0)

        # 融合类型特征和ID特征
        complexes_emb = torch.cat([complexes_type_emb, complexes_id_emb], dim=-1)
        complexes_emb = self.embed_fusion(complexes_emb)

        # 创建各个图的边索引
        complexes_edge_index = get_edges(mask=complexes_mask).cpu()
        complexes_edge_index = torch.LongTensor(complexes_edge_index).to(device)

        ligand_edge_index = get_edges(mask=lig_mask).cpu()
        ligand_edge_index = torch.LongTensor(ligand_edge_index).to(device)

        pocket_edge_index = get_edges(mask=pocket_mask).cpu()
        pocket_edge_index = torch.LongTensor(pocket_edge_index).to(device)

        # 分离复合物嵌入为配体和口袋嵌入
        complexes_emb_ = complexes_emb.clone()
        lig_emb, pocket_emb = complexes_emb_[: num_lig], complexes_emb_[num_lig:]

        # 计算各个图的边特征(距离)
        complexes_distances, _ = coord2diff(complexes_coords, complexes_edge_index)
        if self.sin_embedding is not None:
            complexes_distances = self.sin_embedding(complexes_distances)

        pocket_distances, _ = coord2diff(pocket_coords, pocket_edge_index)
        if self.sin_embedding is not None:
            pocket_distances = self.sin_embedding(pocket_distances)

        lig_distances, _ = coord2diff(lig_coords, ligand_edge_index)
        if self.sin_embedding is not None:
            lig_distances = self.sin_embedding(lig_distances)

        # 通过三个子网络传递特征
        O_C, O_L, O_P = complexes_emb, lig_emb, pocket_emb
        for i in range(self.graph_layers):
            CompLayer = self.ComplexesGraph[i]
            LigLayer = self.LigandGraph[i]
            PocketLayer = self.PocketGraph[i]

            # 更新复合物特征和坐标
            O_C, complexes_coords = CompLayer(O_C, complexes_coords, complexes_edge_index, node_mask=None, edge_mask=None,
                    edge_attr=complexes_distances, update_coords_mask=None)
            
            # 更新配体特征和坐标
            O_L, lig_coords = LigLayer(O_L, lig_coords, ligand_edge_index, node_mask=None, edge_mask=None,
                    edge_attr=lig_distances, update_coords_mask=None)
            
            # 更新口袋特征和坐标
            O_P, pocket_coords = PocketLayer(O_P, pocket_coords, pocket_edge_index, node_mask=None, edge_mask=None,
                    edge_attr=pocket_distances, update_coords_mask=None)

        # 通过融合层整合特征
        FusionLayer = self.FusionGraph[0]

        O_LP = torch.cat([O_L, O_P], dim=0)  # 合并配体和口袋特征
        O_C = FusionLayer(torch.cat([O_C, O_LP], dim=1), complexes_edge_index)  # 融合特征
        
        # 返回配体和口袋的最终特征表示
        return O_C[:num_lig].detach(), O_C[num_lig:].detach()
    

class GCL(nn.Module):
    """
    图卷积层，用于更新节点特征
    
    包含边模型和节点模型，在图神经网络中传递和更新特征
    """
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        """
        初始化图卷积层
        
        参数:
            input_nf: 输入节点特征维度
            output_nf: 输出节点特征维度
            hidden_nf: 隐藏层特征维度
            normalization_factor: 聚合操作的归一化因子
            aggregation_method: 聚合方法('sum'或'mean')
            edges_in_d: 输入边特征维度
            nodes_att_dim: 节点注意力特征维度
            act_fn: 激活函数
            attention: 是否使用注意力机制
        """
        super(GCL, self).__init__()
        input_edge = input_nf * 2  # 边特征维度为源节点和目标节点特征维度之和
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        # 边MLP: 处理边特征
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # 节点MLP: 更新节点特征
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # 可选的注意力层
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        """
        边模型: 计算边特征
        
        参数:
            source: 源节点特征
            target: 目标节点特征
            edge_attr: 初始边特征
            edge_mask: 边掩码
            
        返回:
            处理后的边特征和中间边表示
        """
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        # 应用注意力机制(如果启用)
        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        节点模型: 更新节点特征
        
        参数:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边特征
            node_attr: 额外节点属性
            
        返回:
            更新后的节点特征和聚合特征
        """
        row, col = edge_index
        # 聚合来自邻居的边特征
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                               normalization_factor=self.normalization_factor,
                               aggregation_method=self.aggregation_method)
        
        # 合并节点特征、聚合特征和额外节点属性
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
            
        # 残差连接: 更新节点特征
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        """
        前向传播
        
        参数:
            h: 节点特征
            edge_index: 边索引
            edge_attr: 边特征
            node_attr: 额外节点属性
            node_mask: 节点掩码
            edge_mask: 边掩码
            
        返回:
            更新后的节点特征和边表示
        """
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)  # 计算边特征
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  # 更新节点特征
        if node_mask is not None:
            h = h * node_mask  # 应用节点掩码
        return h, mij


class EquivariantUpdate(nn.Module):
    """
    等变坐标更新层，保持3D坐标的等变性
    
    用于根据节点和边特征更新节点的3D坐标
    """
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        """
        初始化等变坐标更新层
        
        参数:
            hidden_nf: 隐藏层特征维度
            normalization_factor: 聚合操作的归一化因子
            aggregation_method: 聚合方法('sum'或'mean')
            edges_in_d: 边特征维度
            act_fn: 激活函数
            tanh: 是否使用tanh限制更新范围
            coords_range: 坐标更新范围
        """
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d  # 边特征输入维度
        
        # 坐标预测MLP，最后一层初始化较小的权重以稳定训练
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, update_coords_mask=None):
        """
        坐标模型: 更新节点坐标
        
        参数:
            h: 节点特征
            coord: 节点坐标
            edge_index: 边索引
            coord_diff: 坐标差异向量
            edge_attr: 边特征
            edge_mask: 边掩码
            update_coords_mask: 坐标更新掩码
            
        返回:
            更新后的坐标
        """
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)  # 组合源节点、目标节点和边特征
        
        # 预测坐标变换
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range  # 使用tanh约束范围
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)  # 直接预测
            
        if edge_mask is not None:
            trans = trans * edge_mask  # 应用边掩码
            
        # 聚合来自所有相邻节点的坐标变换
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                               normalization_factor=self.normalization_factor,
                               aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg  # 应用坐标更新掩码

        # 更新坐标
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None,
                node_mask=None, edge_mask=None, update_coords_mask=None):
        """
        前向传播
        
        参数:
            h: 节点特征
            coord: 节点坐标
            edge_index: 边索引
            coord_diff: 坐标差异向量
            edge_attr: 边特征
            node_mask: 节点掩码
            edge_mask: 边掩码
            update_coords_mask: 坐标更新掩码
            
        返回:
            更新后的节点坐标
        """
        # 使用坐标模型更新坐标
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask,
                             update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask  # 应用节点掩码
        return coord


class EquivariantBlock(nn.Module):
    """
    等变块，结合图卷积层和等变坐标更新层
    
    同时更新节点特征和3D坐标，保持坐标的等变性
    """
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        """
        初始化等变块
        
        参数:
            hidden_nf: 隐藏层特征维度
            edge_feat_nf: 边特征维度
            act_fn: 激活函数
            n_layers: 图卷积层数量
            attention: 是否使用注意力机制
            norm_diff: 是否归一化坐标差异
            tanh: 是否使用tanh限制坐标更新
            coords_range: 坐标更新范围
            norm_constant: 坐标归一化常数
            sin_embedding: 正弦嵌入对象
            normalization_factor: 聚合操作的归一化因子
            aggregation_method: 聚合方法('sum'或'mean')
        """
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        # 创建多层图卷积层
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                          act_fn=act_fn, attention=attention,
                                          normalization_factor=self.normalization_factor,
                                          aggregation_method=self.aggregation_method))
                                          
        # 创建等变坐标更新层
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                               coords_range=self.coords_range_layer,
                                               normalization_factor=self.normalization_factor,
                                               aggregation_method=self.aggregation_method))

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None, update_coords_mask=None):
        """
        前向传播
        
        参数:
            h: 节点特征
            x: 节点坐标
            edge_index: 边索引
            node_mask: 节点掩码
            edge_mask: 边掩码
            edge_attr: 边特征
            update_coords_mask: 坐标更新掩码
            
        返回:
            更新后的节点特征和坐标
        """
        # 计算坐标差异和边距离
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)  # 使用正弦嵌入表示距离
            
        # 组合边特征
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        
        # 依次通过多层图卷积更新节点特征
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                           node_mask=node_mask, edge_mask=edge_mask)
                                           
        # 通过等变层更新坐标
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr,
                                   node_mask, edge_mask, update_coords_mask=update_coords_mask)

        if node_mask is not None:
            h = h * node_mask  # 应用节点掩码
        return h, x



class SinusoidsEmbeddingNew(nn.Module):
    """
    正弦嵌入层，将标量距离转换为高维特征表示
    
    使用多频率的正弦和余弦函数编码距离信息
    """
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        """
        初始化正弦嵌入层
        
        参数:
            max_res: 最大分辨率
            min_res: 最小分辨率
            div_factor: 频率划分因子
        """
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1  # 计算频率数量
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res  # 生成频率列表
        self.dim = len(self.frequencies) * 2  # 输出维度为频率数量的两倍(sin和cos)

    def forward(self, x):
        """
        前向传播：将标量距离转换为正弦嵌入
        
        参数:
            x: 输入距离值
            
        返回:
            距离的高维正弦嵌入
        """
        x = torch.sqrt(x + 1e-8)  # 对输入取平方根，确保数值稳定
        emb = x * self.frequencies[None, :].to(x.device)  # 乘以频率
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 连接正弦和余弦结果
        return emb.detach()  # 不计算梯度，只作为特征提取器


def coord2diff(x, edge_index, norm_constant=1):
    """
    计算坐标差异和归一化向量
    
    参数:
        x: 节点坐标
        edge_index: 边索引
        norm_constant: 归一化常数
        
    返回:
        radial: 平方半径
        coord_diff: 归一化的坐标差异向量
    """
    row, col = edge_index
    coord_diff = x[row] - x[col]  # 计算坐标差
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)  # 计算平方距离
    norm = torch.sqrt(radial + 1e-8)  # 计算欧几里得距离
    coord_diff = coord_diff/(norm + norm_constant)  # 归一化坐标差异
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """
    对数据按分组进行聚合操作(类似于scatter_add但支持不同的聚合方法)
    
    参数:
        data: 要聚合的数据
        segment_ids: 每个元素的分组ID
        num_segments: 分组总数
        normalization_factor: 归一化因子
        aggregation_method: 聚合方法，'sum'表示求和，'mean'表示平均
        
    返回:
        聚合后的结果
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # 创建全零结果张量
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))  # 扩展分组索引
    result.scatter_add_(0, segment_ids, data)  # 分组求和
    
    # 根据聚合方法进行不同的归一化
    if aggregation_method == 'sum':
        result = result / normalization_factor  # 简单除以归一化因子

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))  # 计算每个分组的元素数量
        norm[norm == 0] = 1  # 防止除零
        result = result / norm  # 计算均值
    return result
