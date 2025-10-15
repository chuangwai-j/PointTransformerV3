"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
from functools import partial
from addict import Dict
import logging
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath
from pointcept.utils.misc import offset2bincount
import warnings
from torch.fx import wrap
from spconv.pytorch import SparseConvTensor

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=16,  # 邻域点数k（原作者用patch_size表示k）
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash

        # 原作者核心参数：邻域点数k=patch_size，前后各取k_half个
        self.patch_size = patch_size
        self.k_half = self.patch_size // 2  # 如k=16→k_half=8

        # 移除flash相关逻辑（因显式邻域无需flash优化）
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    def forward(self, point):
        # 🌟 1.1 获取原作者依赖的核心字段（z-order排序结果）
        sorted_order = point["serialized_order"][self.order_index]  # [N]：排序后的原始点索引
        N = sorted_order.shape[0]  # 当前阶段总点数
        device = sorted_order.device

        # 🌟 1.2 向量化计算“点→排序位置”的映射（快速反向索引）
        sorted_pos = torch.zeros(N, dtype=torch.long, device=device)
        sorted_pos[sorted_order] = torch.arange(N, device=device)  # [N]：每个原始点在排序中的位置

        # 🌟 1.3 实时切片生成邻域（原作者核心逻辑，连续内存访问）
        # 计算每个点的邻域范围（边界裁剪，避免越界）
        start = torch.clamp(sorted_pos - self.k_half, min=0, max=N)  # [N]：邻域起始位置
        end = torch.clamp(sorted_pos + self.k_half + 1, min=0, max=N)  # [N]：邻域结束位置（+1是切片右开区间）
        # 向量化生成邻域位置（0~k-1）
        pos_range = torch.arange(self.patch_size, device=device).unsqueeze(0)  # [1, 16]
        neighbor_pos = start.unsqueeze(1) + pos_range  # [N, 16]：每个点的邻域在sorted_order中的位置
        neighbor_pos = torch.min(neighbor_pos, end.unsqueeze(1) - 1)  # 截断越界位置
        # 提取邻域索引（连续内存访问，GPU极快）
        neighbor_indices = sorted_order[neighbor_pos]  # [N, 16]：最终邻域索引

        # 🌟 1.4 （可选）跨样本校验（如需分样本训练，保留此段；否则可注释）
        if "offset" in point:
            offsets = point["offset"]
            num_samples = len(offsets) - 1
            # 每个点的样本ID
            point_sample_id = torch.searchsorted(offsets[1:], torch.arange(N, device=device))
            # 每个样本的边界
            sample_starts = offsets[:-1][point_sample_id].unsqueeze(1)  # [N, 1]
            sample_ends = offsets[1:][point_sample_id].unsqueeze(1)  # [N, 1]
            # 跨样本掩码：邻域索引超出当前样本范围
            cross_mask = (neighbor_indices < sample_starts) | (neighbor_indices >= sample_ends)
            # 跨样本索引替换为当前点自身（避免干扰）
            self_indices = torch.arange(N, device=device).unsqueeze(1)  # [N, 1]
            neighbor_indices = torch.where(cross_mask, self_indices, neighbor_indices)

        # 🌟 1.5 后续注意力计算（与原逻辑一致，无冗余）
        feat = point.feat  # [N, C]
        qkv = self.qkv(feat)  # [N, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)  # [N, C] ×3

        # 提取邻域的k和v（连续内存访问，快）
        k_neighbor = k[neighbor_indices]  # [N, 16, C]
        v_neighbor = v[neighbor_indices]  # [N, 16, C]

        # 多头维度调整
        H = self.num_heads
        C_head = self.channels // H
        q = q.reshape(N, H, C_head).unsqueeze(2)  # [N, H, 1, C_head]
        k_neighbor = k_neighbor.reshape(N, self.patch_size, H, C_head).permute(0, 2, 1, 3)  # [N, H, 16, C_head]
        v_neighbor = v_neighbor.reshape(N, self.patch_size, H, C_head).permute(0, 2, 1, 3)  # [N, H, 16, C_head]

        # 注意力分数计算
        if self.upcast_attention:
            q = q.float()
            k_neighbor = k_neighbor.float()
        attn = (q * self.scale) @ k_neighbor.transpose(-2, -1)  # [N, H, 1, 16]

        # 可选RPE
        if self.enable_rpe:
            grid_coord = point.grid_coord  # [N, 3]
            neighbor_grid = grid_coord[neighbor_indices]  # [N, 16, 3]
            rel_pos = grid_coord.unsqueeze(1) - neighbor_grid  # [N, 16, 3]
            rpe = self.rpe(rel_pos)  # [N, H, 1, 16]
            attn += rpe

        # 归一化与dropout
        if self.upcast_softmax:
            attn = attn.float()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)

        # 加权求和与投影
        feat_attn = (attn @ v_neighbor).squeeze(2).reshape(N, self.channels)  # [N, C]
        feat_attn = self.proj(feat_attn)
        feat_attn = self.proj_drop(feat_attn)

        # 异常值校验（保留核心，精简日志）
        #if torch.isnan(feat_attn).any() or torch.isinf(feat_attn).any():
        #    sample_paths = point.get('path', ['未知路径'])
        #    logging.error(
        #        f"SerializedAttention异常！样本={sample_paths[:1]}, NaN={torch.isnan(feat_attn).any()}"
        #    )

        # 更新特征
        point.feat = feat_attn
        return point

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # MLP输入数值校验
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error(f"MLP输入异常：含NaN={torch.isnan(x).any().item()}, 含inf={torch.isinf(x).any().item()}")

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size, # 实际被k替代
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        # DropPath（精简实现）
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, point: Point):
        shortcut = point.feat  # 保存原始feat（用于残差连接）

        # 1. CPE层：正常处理Point对象
        point = self.cpe(point)
        point.feat = shortcut + point.feat  # 残差连接
        shortcut = point.feat  # 更新shortcut为CPE处理后的feat

        # 2. 注意力层 + DropPath：手动处理Point对象，不破坏结构
        if self.pre_norm:
            point = self.norm1(point)
        # 关键修改：先获取attn处理后的Point对象，再单独对feat应用drop_path
        point = self.attn(point)  # 得到Point对象
        point.feat = shortcut + self.drop_path(point.feat)

        # 3. MLP层 + DropPath：同样手动处理，保留Point对象
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + self.drop_path(point.feat)
        '''
        # 关键修改：先获取mlp处理后的Point对象，再对feat应用drop_path
        point_mlp = self.mlp(point)  # 得到Point对象
        point_mlp = self.mlp_norm(point_mlp)  # 新增：稳定MLP层输出
        point_mlp.feat = self.drop_path(point_mlp.feat)
        # 残差连接
        point_mlp.feat = shortcut + point_mlp.feat
        point = point_mlp
        if not self.pre_norm:
            point = self.norm2(point)
        '''

        # 4. 更新sparse_conv_feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point  # 确保返回的是Point对象


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer else None
        self.act = PointSequential(act_layer()) if act_layer else None

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code", "serialized_order", "serialized_inverse", "serialized_depth"
        }.issubset(point.keys()), "需先调用serialization()"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        pooled_batch = point.batch[head_indices]  # 池化后每个点的样本ID（0~num_samples-1）
        new_total_points = len(head_indices)  # 池化后的总点数（关键：由聚类结果决定）

        # 🌟 修复：基于池化后的batch统计每个样本的实际点数，生成正确的new_offset
        num_samples = len(point.offset) - 1  # 样本数量不变
        # 统计每个样本在池化后的点数（bincount：索引为样本ID，值为该样本的点数）
        downsampled_bincount = torch.bincount(pooled_batch, minlength=num_samples)
        # 生成新offset（累加实际点数）
        new_offset = torch.cat([
            torch.tensor([0], device=point.offset.device),
            torch.cumsum(downsampled_bincount, dim=0)
        ], dim=0)
        assert new_offset[-1].item() == new_total_points, "Pooling后offset错误"

        # generate down code, order, inverse生成池化后的排序相关字段
        code = code[:, head_indices]
        order = torch.argsort(code)
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        # collect information 构建新Point对象
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            offset=new_offset,  # 关键：添加下采样后的正确offset
            path = point.get('path', ['未知路径'])  # 🌟 新增：保留样本路径，用于异常定位
        )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)

        # 精简校验（保留核心，减少日志）
        #if torch.isnan(point.feat).any() or torch.isinf(point.feat).any():
        #    logging.error(f"SerializedPooling异常！样本={point['path'][:1]}")

        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys() and "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")

        # 精简校验
        if torch.isnan(point.feat).any() or torch.isinf(parent.feat).any():
            logging.error(f"SerializedUnpooling异常！")

        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        if torch.isnan(point.feat).any() or torch.isinf(point.feat).any():
            logging.error(f"Embedding输入异常！样本={point['path'][:1]}")

        # 嵌入层处理（可能创建新Point对象）
        point = self.stem(point)

        # 精简校验
        if torch.isnan(point.feat).any() or torch.isinf(point.feat).any():
            logging.error(f"Embedding输出异常！样本={point['path'][:1]}")

        return point


@MODELS.register_module('PT-v3m1')
class PointTransformerV3(PointModule):
    def __init__(
        self,
        num_classes=5,  # 明确为5分类（0-4）
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2),
        enc_depths=(1, 1, 3, 1),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(16, 16, 16, 16),  # 与k_neighbors=16保持一致
        dec_depths=(1, 1, 1),
        dec_channels=(64, 64, 128),
        dec_num_head=(4, 4, 8),
        dec_patch_size=(16, 16, 16),   # 与k_neighbors=16保持一致
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,  # 修正：用户未安装flash_attn，设为False
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_classes = num_classes  # 保存类别数
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        # 校验参数长度（确保编码器/解码器参数匹配）
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths) == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)
        if not self.cls_mode:
            assert self.num_stages == len(dec_depths) + 1 == len(dec_channels) + 1 == len(dec_num_head) + 1 == len(
                dec_patch_size) + 1

        # 归一化层配置
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        # 嵌入层
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # 编码器
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],   # 传递正确的邻域点数
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # 解码器
        self.dec = None
        self.original_dec_channels = dec_channels  # 保存原始解码器通道配置
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            # 注意：这里拼接了编码器最后一层通道，但仅用于解码器内部计算
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1][::-1])
                ]
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        # 分类头（多分类，0-4共5类）
        '''
        if not self.cls_mode:
            self.head = nn.Linear(self.original_dec_channels[0], self.num_classes)  # 输出5个通道（对应5类）
        else:
            self.head = nn.Linear(enc_channels[-1], self.num_classes)  # 输出5个通道（对应5类）
        '''
        if not self.cls_mode:
            in_channels = self.original_dec_channels[0]
        else:
            in_channels = enc_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(in_channels, 256),  # 第一层线性变换，拓宽维度
            nn.ReLU(inplace=True),  # 激活函数，增加非线性表达
            nn.Dropout(0.3),  # 随机失活30%神经元，防止过拟合
            nn.Linear(256, self.num_classes)  # 第二层线性变换，输出5类预测
        )

    def forward(self, data_dict):
        #  首先检查path是否存在且有效
        #if 'path' not in data_dict or data_dict['path'][0] == '未知路径':
        #    raise ValueError(f"样本path丢失！当前data_dict中的path: {data_dict.get('path', '无')}")
        #  关键1：保留样本路径，用于异常定位
        sample_paths = data_dict.get('path', ['未知路径'])
        #  关键2：计算并打印spatial_shape（验证集核心调试信息）
        coord = data_dict['coord']
        spatial_shape = [
            int(coord[:, 2].max().item()) + 1,  # z轴（spconv默认z/y/x顺序，必须对应）
            int(coord[:, 1].max().item()) + 1,  # y轴
            int(coord[:, 0].max().item()) + 1  # x轴（注意：coord是[x,y,z]，需调整顺序）
        ]
        # 区分训练/验证模式，打印spatial_shape（关键：验证是否过大）
        mode = "训练集" if self.training else "验证集"
        logging.info(
            f"【{mode}】样本路径={[os.path.basename(p) for p in sample_paths[:2]]}, "
            f"总点数={coord.shape[0]}, spatial_shape={spatial_shape}（z/y/x）"
        )
        # 检查spatial_shape是否过大（超过2000视为异常，需后续裁剪coord）
        #TODO
        '''
        if any(dim > 2000 for dim in spatial_shape):
            logging.warning(
                f"⚠️ {mode} spatial_shape过大！各维度应≤2000，当前={spatial_shape}，可能导致logits=nan"
            )
        '''
        # 1. 构建Point对象（保留path字段）
        data_dict['path'] = sample_paths  # 确保path传入Point对象
        point = Point(data_dict)
        # 2. 保留序列化逻辑（原有代码，处理点云顺序）
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)

        # 🌟 新增：提取排序邻域索引（核心修改1）
        # 注意：k值需与模型原k邻域一致（如16），避免后续层输入维度不匹配
        k_neighbors = 16  # 可根据模型实际需求调整（必须为偶数）
        point.get_sorted_neighbors(k=k_neighbors)

        # 🌟 关键3：替换point.sparsify()，手动构建SparseConvTensor（带allow_empty=True）
        # 3.1 生成样本索引（batch_idx）：每个点属于哪个样本
        batch_size = len(point.offset) - 1  # offset长度=样本数+1，如[0,1920,3840]对应2个样本
        batch_idx = []
        for i in range(batch_size):
            start = point.offset[i].item()  # 第i个样本的起始点索引
            end = point.offset[i + 1].item()  # 第i个样本的结束点索引
            batch_idx.extend([i] * (end - start))  # 为每个点分配样本索引
        # 转换为张量并调整形状（[N,1]，N为总点数）
        batch_idx = torch.tensor(batch_idx, device=point.coord.device, dtype=torch.int32).unsqueeze(1)

        # 3.2 构建indices（spconv要求格式：[z, y, x, batch_idx]，共4列）
        # 注意：coord原始格式是[x,y,z]，需调整为[z,y,x]
        z_coord = point.coord[:, 2].unsqueeze(1).to(torch.int32)  # 第3列是z
        y_coord = point.coord[:, 1].unsqueeze(1).to(torch.int32)  # 第2列是y
        x_coord = point.coord[:, 0].unsqueeze(1).to(torch.int32)  # 第1列是x
        indices = torch.cat([z_coord, y_coord, x_coord, batch_idx], dim=1)  # 拼接为[N,4]

        # 3.3 手动创建SparseConvTensor，显式设置allow_empty=True
        # 步骤1：先创建空的SparseConvTensor（用默认参数）
        sparse_tensor = spconv.SparseConvTensor(
            features=point.feat,  # 点云特征（[N, C]）
            indices=indices,  # 坐标+样本索引（[N,4]）
            spatial_shape=spatial_shape,  # 体素网格大小（z/y/x）
            batch_size=batch_size,  # 批次大小
        )
        # 步骤2：手动设置allow_empty=True（绕过fx对__init__的追踪）
        sparse_tensor.allow_empty = True  # 直接修改属性，而非通过__init__参数

        # 赋值给point.sparse_conv_feat
        point.sparse_conv_feat = sparse_tensor

        # 4.嵌入层
        point = self.embedding(point)

        # 5. 编码器（无需重新生成邻域）
        point = self.enc(point)

        # 6.解码器（分割模式）
        if not self.cls_mode and self.dec is not None:
            point = self.dec(point)

        # 7.分类头计算logits与数值校验
        logits = self.head(point.feat)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.error(f"【{mode}】logits异常！样本={sample_paths[:1]}")
        else:
            logging.info(f"【{mode}】logits正常！形状={logits.shape}")

        return logits  # 返回对象，而非张量
