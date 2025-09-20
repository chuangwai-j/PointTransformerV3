"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

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

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
#from pointcept.models.utils.misc import offset2bincount
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
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
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
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"

        # 仅在首次调用时计算，避免重复计算
        if (pad_key not in point.keys() or
                unpad_key not in point.keys() or
                cu_seqlens_key not in point.keys()):

            offset = point.offset
            device = offset.device
            bincount = offset2bincount(point.offset,check_padding=False)  # 每个样本的点数
            total_original_points = offset[-1].item()  # 当前batch总点数
            logging.debug(f"【get_padding】总点数={total_original_points}, unpad长度将设为={total_original_points}")

            # 1. 计算需要padding到的点数（确保是patch_size的倍数）
            bincount_pad = (
                    torch.div(bincount + self.patch_size - 1,
                              self.patch_size,
                              rounding_mode="trunc") * self.patch_size
            )
            # 只对点数超过patch_size的样本进行padding
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

            # 2. 计算偏移量（带padding和不带padding的）
            _offset = nn.functional.pad(offset, (1, 0))  # 原始偏移量（前补0）
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))  # 带padding的偏移量


            # 3. 初始化pad和unpad数组（确保类型为长整数，避免索引错误）
            total_padded_points = _offset_pad[-1].item()  # 带padding的总点数
            #pad = torch.arange(total_padded_points, device=device, dtype=torch.long)
            #unpad = torch.arange(total_original_points, device=device, dtype=torch.long)
            # 修复：unpad长度必须等于总点数，避免长度不足
            unpad = torch.zeros(total_original_points, dtype=torch.long, device=device)
            pad = torch.arange(total_padded_points, device=device, dtype=torch.long)

            # 4. 计算cu_seqlens（用于Flash Attention的序列长度索引）
            cu_seqlens = []
            for i in range(len(offset) - 1):
                orig_start, orig_end = offset[i], offset[i + 1]  # 原始样本范围
                pad_start, pad_end = _offset_pad[i], _offset_pad[i + 1]  # padding后样本范围
                sample_points_orig = orig_end - orig_start
                sample_points_pad = pad_end - pad_start

                # 3.1 校正unpad：映射原始索引→padding后索引（确保不超界）
                if sample_points_orig > 0:
                    offset_val = pad_start - orig_start
                    # 限制unpad不超过total_original_points
                    unpad_slice = unpad[orig_start:orig_end] + offset_val
                    unpad[orig_start:orig_end] = torch.clamp(unpad_slice, 0, total_original_points - 1)

                # 3.2 校正pad：处理padding区域（避免复制越界）
                if sample_points_orig != sample_points_pad and sample_points_pad > 0:
                    # 安全计算复制源范围（避免src超出原始样本）
                    src_start = max(orig_start, orig_end - self.patch_size)  # 取原始样本最后patch_size个点
                    src_end = orig_end
                    src_len = src_end - src_start

                    # 安全计算复制目标范围
                    dst_start = orig_end
                    dst_end = pad_end
                    dst_len = dst_end - dst_start

                    # 仅当源和目标都有效时复制
                    if src_len > 0 and dst_len > 0:
                        copy_len = min(src_len, dst_len)
                        # 映射src到padding后的pad索引
                        pad[dst_start:dst_start + copy_len] = pad[src_start:src_start + copy_len]

                # 3.3 校正pad的样本内偏移（确保与原始索引对齐）
                if sample_points_pad > 0:
                    pad_slice = pad[pad_start:pad_end] - (pad_start - orig_start)
                    pad[pad_start:pad_end] = torch.clamp(pad_slice, 0, total_original_points - 1)

                # 3.4 生成cu_seqlens（确保步长合理）
                step = max(1, self.patch_size)  # 避免步长为0
                seq = torch.arange(pad_start, pad_end, step, dtype=torch.int32, device=device)
                # 确保序列覆盖到pad_end
                if len(seq) == 0 or seq[-1] < pad_end - 1:
                    seq = torch.cat([seq, torch.tensor([pad_end - 1], device=device, dtype=torch.int32)])
                cu_seqlens.append(seq)

                # 4. 合并cu_seqlens（确保最后一个元素是总长度）
            merged_cu_seqlens = torch.cat(cu_seqlens) if cu_seqlens else torch.tensor([0], device=device,
                                                                                      dtype=torch.int32)
            if merged_cu_seqlens[-1] != total_padded_points:
                merged_cu_seqlens = torch.cat(
                    [merged_cu_seqlens, torch.tensor([total_padded_points], device=device, dtype=torch.int32)])

            # 最终校验unpad的有效性
            if (unpad >= total_original_points).any() or (unpad < 0).any():
                invalid = unpad[(unpad >= total_original_points) | (unpad < 0)]
                warnings.warn(f"unpad中存在无效索引：{invalid[:5]}（前5个），已自动截断")
                unpad = torch.clamp(unpad, 0, total_original_points - 1)

            # 保存结果
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = merged_cu_seqlens

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        # 在point.serialization前添加字段检查
        logging.debug(
            f"【Point对象字段检查】keys={point.keys()}, 包含grid_size={('grid_size' in point.keys())}, 包含coord={('coord' in point.keys())}")

        if not self.enable_flash:
            # 计算每个样本的点数（bincount）
            logging.debug(f"\n【调试日志】当前batch的offset: {point.offset}")
            bincount = offset2bincount(point.offset, check_padding=False)
            logging.debug(f"【调试日志】计算出的样本点数bincount: {bincount}")

            # 确保样本点数有效
            min_points_per_sample = bincount.min().item()
            if min_points_per_sample <= 0:
                raise ValueError(f"样本点数异常：存在点数≤0的样本（min_points={min_points_per_sample}）")

            # 计算合理的patch_size
            self.patch_size = min(min_points_per_sample, self.patch_size_max)
            self.patch_size = max(self.patch_size, 1)  # 避免patch_size为0

        H = self.num_heads
        K = self.patch_size
        logging.debug(
            f"当前patch_size (K): {K}, 配置patch_size_max: {self.patch_size_max}, 样本最小点数: {min_points_per_sample}")
        assert K >= 1, f"patch_size (K) 必须≥1，当前K={K}"
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # ====================== 新增：serialized_order索引校验 ======================
        # 1. 获取当前order_index对应的serialized_order切片
        serialized_order_slice = point.serialized_order[self.order_index]
        serialized_order_len = serialized_order_slice.shape[0]

        # 2. 校验pad的索引是否超出serialized_order_slice的长度
        if (pad < 0).any() or (pad >= serialized_order_len).any():
            # 截断超界的pad索引（兜底，避免直接触发CUDA断言）
            pad = torch.clamp(pad, 0, serialized_order_len - 1)
            # 打印警告日志，定位问题来源
            logging.debug(
                f"pad索引超出serialized_order范围！serialized_order长度={serialized_order_len}，"
                f"校正前pad范围: [{pad.min()}, {pad.max()}]，校正后范围: [0, {serialized_order_len - 1}]"
            )

        # 3. 执行索引操作（此时pad已确保在有效范围）
        order = serialized_order_slice[pad]
        # =====================================================================

        # ====================== 新增：serialized_inverse索引校验 ======================
        # 1. 先获取serialized_inverse索引
        serialized_inverse_idx = point.serialized_inverse[self.order_index]
        unpad_len = unpad.shape[0]  # unpad的长度（有效索引范围：0 ~ unpad_len-1）

        # 2. 校验serialized_inverse_idx是否超出unpad的有效范围
        if (serialized_inverse_idx < 0).any() or (serialized_inverse_idx >= unpad_len).any():
            # 截断超界索引（兜底，避免直接报错）
            serialized_inverse_idx = torch.clamp(serialized_inverse_idx, 0, unpad_len - 1)
            # 打印警告，定位问题样本
            logging.debug(
                f"serialized_inverse索引超出unpad范围！unpad长度={unpad_len}，"
                f"校正前最大索引={serialized_inverse_idx.max()}, 最小索引={serialized_inverse_idx.min()}"
            )

        # 3. 执行unpad映射（此时索引已确保在有效范围）
        inverse = unpad[serialized_inverse_idx]
        # =====================================================================

        # ====================== 新增：样本内索引最终校正 ======================
        feat_len = point.feat.shape[0]
        offset = point.offset
        # 核心修复：从point.feat获取设备（确保与输入数据在同一设备）
        device = point.feat.device
        # 生成每个点的样本归属标记
        sample_id = torch.zeros(feat_len, dtype=torch.int64, device=device)
        for i in range(1, len(offset)):
            sample_id[offset[i - 1]:offset[i]] = i - 1

        # 逐样本校正inverse：确保每个样本的inverse在自身范围内
        for i in range(len(offset) - 1):
            sample_mask = (sample_id == i)
            sample_inverse = inverse[sample_mask]
            # 样本内有效范围：[offset[i], offset[i+1])
            valid_min = offset[i]
            valid_max = offset[i + 1] - 1
            # 截断超界索引（兜底）
            sample_inverse_clamped = torch.clamp(sample_inverse, valid_min, valid_max)
            inverse[sample_mask] = sample_inverse_clamped
        # =====================================================================

        # ====================== 新增：inverse索引关键校验 ======================
        # 1. 检查inverse与feat的长度匹配
        feat_len = point.feat.shape[0]
        inverse_len = inverse.shape[0]
        if inverse_len != feat_len:
            raise ValueError(
                f"inverse长度与feat不匹配！"
                f"inverse长度={inverse_len}, feat点数={feat_len}"
            )

        # 2. 检查索引范围（核心修复）
        invalid_mask = (inverse < 0) | (inverse >= feat_len)
        if invalid_mask.any():
            # 收集无效索引信息
            invalid_indices = inverse[invalid_mask]
            first_invalid = invalid_indices[:5]  # 取前5个示例
            invalid_count = invalid_indices.numel()
            # 打印样本边界辅助调试
            sample_ranges = [f"样本{i}: [{point.offset[i]}, {point.offset[i + 1]})"
                             for i in range(len(point.offset) - 1)]
            raise ValueError(
                f"inverse索引越界！有效范围应在[0, {feat_len})，"
                f"共发现{invalid_count}个无效索引，示例: {first_invalid}\n"
                f"样本边界: {sample_ranges}"
            )
        # =====================================================================

        # 样本点数与注意力头数匹配检查
        logging.debug("=" * 50)
        logging.debug(f"当前注意力头数H: {self.num_heads}")
        logging.debug(f"offset: {point['offset']}")
        sample_points = [point['offset'][i + 1] - point['offset'][i] for i in range(len(point['offset']) - 1)]
        logging.debug(f"每个样本的点数: {sample_points}")
        for i, sp in enumerate(sample_points):
            if sp % self.num_heads != 0:
                logging.debug(f"❌ 样本{i}点数{sp}不能被头数{self.num_heads}整除！K={sp // self.num_heads}")
            else:
                logging.debug(f"✅ 样本{i}点数{sp}，K={sp // self.num_heads}")

        # 特征维度一致性检查
        logging.debug(f"coord点数: {point['coord'].shape[0]}")
        logging.debug(f"feat点数: {point['feat'].shape[0]}")
        logging.debug(f"label点数: {point['label'].shape[0]}")
        logging.debug(f"beamaz点数: {point['beamaz'].shape[0] if 'beamaz' in point else '无'}")
        logging.debug(f"inverse形状: {inverse.shape}, 最大索引: {inverse.max()}, 最小索引: {inverse.min()}")
        logging.debug("=" * 50)

        # 注意力计算逻辑
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)

        # 使用经过校验的inverse索引
        feat = feat[inverse]

        # 后续处理
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        logging.debug(f"SerializedAttention输出point.feat形状: {point.feat.shape}")
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
        enable_flash=True,
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
            patch_size=patch_size,
            num_heads=num_heads,
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
        '''self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )'''
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, point: Point):
        logging.debug(f"Block输入point类型: {type(point)}")  # 应输出 <class 'pointcept.models.utils.structure.Point'>
        shortcut = point.feat  # 保存原始feat（用于残差连接）
        # 1. CPE层：正常处理Point对象
        point = self.cpe(point)
        point.feat = shortcut + point.feat  # 残差连接
        shortcut = point.feat  # 更新shortcut为CPE处理后的feat

        # 2. 注意力层 + DropPath：手动处理Point对象，不破坏结构
        if self.pre_norm:
            point = self.norm1(point)
        # 关键修改：先获取attn处理后的Point对象，再单独对feat应用drop_path
        point_attn = self.attn(point)  # 得到Point对象
        # 只对feat应用drop_path，保留Point对象其他字段
        point_attn.feat = self.drop_path(point_attn.feat)
        # 残差连接：更新feat
        point_attn.feat = shortcut + point_attn.feat
        # 传递更新后的Point对象
        point = point_attn
        if not self.pre_norm:
            point = self.norm1(point)

        # 3. MLP层 + DropPath：同样手动处理，保留Point对象
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        # 关键修改：先获取mlp处理后的Point对象，再对feat应用drop_path
        point_mlp = self.mlp(point)  # 得到Point对象
        point_mlp.feat = self.drop_path(point_mlp.feat)
        # 残差连接
        point_mlp.feat = shortcut + point_mlp.feat
        point = point_mlp
        if not self.pre_norm:
            point = self.norm2(point)

        # 4. 更新sparse_conv_feat（原逻辑不变）
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
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        # -------------------------- 新增：计算下采样后的offset --------------------------
        # 1. 从原始point获取正确的offset，计算原始样本点数
        orig_offset = point.offset  # 原始offset（如[0,1920,3840]）
        orig_bincount = offset2bincount(orig_offset, check_padding=False)  # 原始样本点数（如[1920,1920]）
        # 2. 计算下采样后的每个样本点数（原始点数 // 下采样比例stride）
        downsampled_bincount = orig_bincount // self.stride  # 如[960,960]（stride=2时）
        # 3. 生成下采样后的新offset（累加下采样后的点数）
        new_offset = torch.cat(
            [torch.tensor([0], device=orig_offset.device),
             torch.cumsum(downsampled_bincount, dim=0)],
            dim=0
        )  # 新offset如[0,960,1920]
        # -------------------------- 新增结束 --------------------------

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
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
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
            offset=new_offset  # 关键：添加下采样后的正确offset
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
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
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
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
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module('PT-v3m1')
class PointTransformerV3(PointModule):
    def __init__(
        self,
        # 新增：添加 num_classes 参数（默认2，适配你的二分类）
        num_classes=2,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
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
        # 保存 num_classes 到实例变量
        #self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths) == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)
        if not self.cls_mode:
            assert self.num_stages == len(dec_depths) + 1 == len(dec_channels) + 1 == len(dec_num_head) + 1 == len(
                dec_patch_size) + 1

        # norm layers
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

        # 嵌入层（
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
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
                        patch_size=enc_patch_size[s],
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

        # decoder解码器
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
                #dec_drop_path_.reverse()
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

        # 新增：分类头（在解码器之后，将特征映射到类别数）
        # 注意：这部分是新增的，放在解码器代码后面，而非替换
        if not self.cls_mode:
            # 解码器最后一个阶段的输出通道数是 dec_channels[0]
            self.head = nn.Linear(self.original_dec_channels[0], 1)
        else:
            # 分类模式下使用编码器最后一层的通道数
            self.head = nn.Linear(enc_channels[-1], 1)

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode and self.dec is not None:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )

        # 新增：通过分类头输出预测结果
        # 风切变检测是“点级预测”，每个点输出一个二分类结果（0/1）
        #point.feat = self.head(point.feat)  # (N_points, num_classes)
        logits = self.head(point.feat)
        return logits  # 返回Point对象，而非张量
