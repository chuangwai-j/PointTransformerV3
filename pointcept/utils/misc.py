#pointcept/utils/misc.py
"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import logging
import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


'''def collate_fn(batch):
    """
    融合版批处理函数：
    1. 保留原函数对grid_size等特殊字段的处理
    2. 加入beamaz的维度校验，确保与coord严格同步
    """
    if not batch:
        return {}

    if not isinstance(batch[0], dict):
        return torch.utils.data.dataloader.default_collate(batch)

    result = {}
    offsets = [0]
    # 先计算offset并收集beamaz用于校验
    beamaz_list = []
    for item in batch:
        num_points = len(item['coord']) if 'coord' in item else 0
        offsets.append(offsets[-1] + num_points)

        # 🌟 单样本内beamaz与coord维度校验
        if 'beamaz' in item:
            assert len(item['beamaz']) == num_points, \
                f"样本{item.get('path', '未知')}中beamaz长度({len(item['beamaz'])})与coord点数({num_points})不一致"
            beamaz_list.append(item['beamaz'])

    result['offset'] = torch.tensor(offsets, dtype=torch.int64)

    # 处理其他字段
    for key in batch[0].keys():
        if key == 'offset':
            continue

        values = [item[key] for item in batch]

        # 保留grid_size的特殊处理（原函数功能）
        if key == 'grid_size':
            result[key] = values[0].clone().detach().float() if isinstance(values[0], torch.Tensor) \
                else torch.tensor(values[0], dtype=torch.float32)
            continue

        # 处理张量/数组类型
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.cat(values, dim=0)
        elif hasattr(values[0], '__array__'):
            if key in ['coord', 'feat', 'beamaz']:
                result[key] = torch.cat([torch.from_numpy(v).float() for v in values], dim=0)
            elif key == 'label':
                result[key] = torch.cat([torch.from_numpy(v).long() for v in values], dim=0)
            else:
                result[key] = values
                print(f"Warning: Could not concatenate {key}, keeping as list")
        else:
            result[key] = values

    # 🌟 批次级beamaz与coord维度校验
    if 'beamaz' in result:
        assert len(result['beamaz']) == len(result['coord']), \
            f"批次内beamaz总长度({len(result['beamaz'])})与coord总长度({len(result['coord'])})不一致"


    return result
'''


def collate_fn(batch):
    """
    适配风切变数据的批处理函数：
    1. 过滤空样本，避免点数≤0；2. 校验补点逻辑（点数为384的倍数）；
    3. 正确区分字段维度（coord/feat为2维，label/beamaz为1维）；4. 确保offset严格递增
    """

    # -------------------------- 新增：第一步过滤None样本 --------------------------
    # 先移除__getitem__返回的None（采样点数不足的样本）
    batch = [item for item in batch if item is not None]
    if not batch:
        warnings.warn("当前batch所有样本均为无效（点数不足），返回空batch，需在训练循环中跳过")
        return None  # 返回空，训练循环中处理
    # --------------------------------------------------------------------------

    # 1. 过滤无效/空样本
    valid_batch = []
    for idx, item in enumerate(batch):
        if not isinstance(item, dict) or 'coord' not in item:
            warnings.warn(f"过滤无效样本{idx}：非dict或缺失coord字段", UserWarning)
            continue
        # 统一点数计算逻辑（适配Tensor/numpy数组）
        num_points = item['coord'].shape[0] if isinstance(item['coord'], torch.Tensor) else len(item['coord'])
        if num_points <= 0:
            warnings.warn(f"过滤空样本{idx}：点数={num_points}", UserWarning)
            continue
        valid_batch.append(item)
    if not valid_batch:
        raise ValueError("当前batch无有效样本！请检查数据预处理/补点流程")

    # 2. 初始化变量，校验单样本合法性
    result = {}
    offsets = [0]
    total_points = 0
    sample_sizes = []
    device = valid_batch[0]['coord'].device if isinstance(valid_batch[0]['coord'], torch.Tensor) else torch.device(
        'cpu')

    for idx, item in enumerate(valid_batch):
        # 2.1 校验点数是否为384的倍数（适配补点逻辑）
        num_points = item['coord'].shape[0] if isinstance(item['coord'], torch.Tensor) else len(item['coord'])
        if num_points % 384 != 0:
            raise ValueError(
                f"样本{idx}点数异常：{num_points}（需为384的倍数，如1536/1920）\n"
                "请检查补点代码是否正常执行"
            )
        sample_sizes.append(num_points)
        total_points += num_points
        offsets.append(total_points)

        # 2.2 统一字段类型+维度校验（核心修改：区分字段维度要求）
        for key in ['coord', 'feat', 'label', 'beamaz', 'grid_size']:
            if key not in item:
                raise KeyError(f"样本{idx}缺失必要字段：{key}")

            # numpy转Tensor，统一类型
            if isinstance(item[key], np.ndarray):
                if key == 'label':
                    dtype = torch.long
                elif key in ['coord', 'feat', 'beamaz', 'grid_size']:
                    dtype = torch.float32
                else:
                    dtype = torch.float32
                item[key] = torch.from_numpy(item[key]).to(dtype).to(device)
            elif not isinstance(item[key], torch.Tensor):
                raise TypeError(f"样本{idx}的{key}类型异常：需Tensor/numpy数组")

            # -------------------------- 核心修改：维度校验逻辑 --------------------------
            if key == 'coord':
                # coord：2维 (N, 3)，N=点数，3=x/y/z
                if item[key].dim() != 2 or item[key].shape[0] != num_points or item[key].shape[1] != 3:
                    raise ValueError(
                        f"样本{idx}的coord异常：维度={item[key].dim()}，形状={item[key].shape}\n"
                        f"需为2维张量 (点数, 3)，当前点数={num_points}，应满足形状=({num_points}, 3)"
                    )
            elif key == 'feat':
                # feat：2维 (N, C)，N=点数，C=特征维度（如9）
                if item[key].dim() != 2 or item[key].shape[0] != num_points:
                    raise ValueError(
                        f"样本{idx}的feat异常：维度={item[key].dim()}，形状={item[key].shape}\n"
                        f"需为2维张量 (点数, 特征维度)，当前点数={num_points}，应满足形状=({num_points}, C)（如C=9）"
                    )
            elif key in ['label', 'beamaz']:
                # label/beamaz：1维 (N,)，N=点数
                if item[key].dim() != 1 or item[key].shape[0] != num_points:
                    raise ValueError(
                        f"样本{idx}的{key}异常：维度={item[key].dim()}，形状={item[key].shape}\n"
                        f"需为1维张量 (点数,)，当前点数={num_points}，应满足形状=({num_points},)"
                    )
            elif key == 'grid_size':
                # grid_size：1维 (3,)，3=x/y/z网格大小
                if item[key].dim() != 1 or item[key].shape[0] != 3:
                    raise ValueError(f"样本{idx}的grid_size异常：需1维张量 (3,)，实际形状={item[key].shape}")
            # --------------------------------------------------------------------------

    # 3. 生成offset并校验
    result['offset'] = torch.tensor(offsets, dtype=torch.int64, device=device)
    if not (torch.diff(result['offset']) > 0).all():
        raise ValueError(f"offset生成异常（需严格递增）：{result['offset'].tolist()}")
    if result['offset'][-1] != total_points:
        raise ValueError(
            f"offset总点数不匹配：offset[-1]={result['offset'][-1]}，实际总点数={total_points}"
        )

    # 4. 拼接点级字段（正确处理2维/1维张量）
    # coord/feat：2维 (N_total, 3) / (N_total, C)，按dim=0拼接
    result['coord'] = torch.cat([item['coord'] for item in valid_batch], dim=0)
    result['feat'] = torch.cat([item['feat'] for item in valid_batch], dim=0)
    # label/beamaz：1维 (N_total,)，按dim=0拼接
    result['label'] = torch.cat([item['label'] for item in valid_batch], dim=0)
    result['beamaz'] = torch.cat([item['beamaz'] for item in valid_batch], dim=0)

    # 5. 校验拼接后维度
    assert result['coord'].shape == (total_points, 3), f"coord拼接异常：{result['coord'].shape} != ({total_points}, 3)"
    assert result['feat'].shape[0] == total_points, f"feat拼接异常：{result['feat'].shape[0]} != {total_points}"
    assert result['label'].shape == (total_points,), f"label拼接异常：{result['label'].shape} != ({total_points},)"
    assert result['beamaz'].shape == (total_points,), f"beamaz拼接异常：{result['beamaz'].shape} != ({total_points},)"

    # 6. 处理grid_size（支持同/不同样本场景）
    # 6. 处理grid_size（核心修改：强制转为张量，避免列表类型）
    grid_sizes = [item['grid_size'] for item in valid_batch]
    # 方案：无论样本间是否一致，均取第一个样本的grid_size（确保为张量类型，适配Point对象）
    # 理由：grid_size是预处理的网格参数，单batch内差异对模型影响极小，优先保证字段有效性
    result['grid_size'] = grid_sizes[0].clone().detach().float()
    # 移除之前的“列表保留逻辑”，避免grid_size为列表
    # （可选）打印警告，提示样本间grid_size差异
    if not all(torch.equal(gs, result['grid_size']) for gs in grid_sizes):
        logging.debug(
            f"当前batch样本grid_size不一致（已取第一个样本的{result['grid_size']}作为统一值）\n"
            f"各样本grid_size：{[gs.tolist() for gs in grid_sizes]}"
        )

    # 7. 保留path字段（字符串列表）
    if 'path' in valid_batch[0]:
        result['path'] = [item['path'] for item in valid_batch]

    # 8. 调试日志
    logging.info(f"✅ Batch生成成功：样本数={len(valid_batch)}，总点数={total_points}")
    logging.info(f"   各样本点数：{sample_sizes}（均为384的倍数）")
    logging.info(f"   拼接后维度：coord={result['coord'].shape}，feat={result['feat'].shape}，label={result['label'].shape}")
    logging.info(f"   Offset：{result['offset'].tolist()}")

    return result


'''
def offset2bincount(offset):
    """
    将offset转换为“每个样本的点数”（模型可能需要用这个计算单样本损失）
    例：offset=[0,2603,8235] → bincount=[2603, 5632]
    """
    if len(offset) < 2:
        return torch.tensor([0], dtype=torch.int64)
    # 计算相邻offset的差值，即每个样本的点数
    bincount = offset[1:] - offset[:-1]
    return bincount
'''

def offset2bincount(offset, check_padding=True):
    """
    从offset计算样本点数（增强鲁棒性）：
    - 校验offset合法性；- 确保无点数≤0的样本；- 可选校验补点逻辑（仅对原始样本）
    参数:
        check_padding: 是否校验补点逻辑（原始样本设为True，下采样后中间样本设为False）
    """
    # 1. 基础类型/维度校验
    if not isinstance(offset, torch.Tensor):
        raise TypeError(f"offset必须为torch.Tensor，实际类型：{type(offset)}")
    if offset.dim() != 1:
        raise ValueError(f"offset必须为1维张量，实际维度：{offset.dim()}")
    if offset.shape[0] < 2:
        raise ValueError(f"offset长度必须≥2（如[0, 1536]），实际长度：{offset.shape[0]}")

    # 2. 计算样本点数，校验有效性（核心必要校验，任何阶段都需通过）
    bincount = offset[1:] - offset[:-1]
    # 定位点数≤0的样本（任何情况下都不允许）
    invalid_mask = bincount <= 0
    if invalid_mask.any():
        invalid_indices = torch.where(invalid_mask)[0].tolist()
        invalid_values = bincount[invalid_mask].tolist()
        raise ValueError(
            f"存在点数≤0的样本：索引={invalid_indices}，点数={invalid_values}\n"
            f"完整offset：{offset.tolist()}，完整样本点数：{bincount.tolist()}"
        )

    # 3. 可选：校验补点逻辑（仅对原始输入样本生效，下采样后样本跳过）
    if check_padding:
        # 原始样本必须满足≥1536（补点后的最小要求）
        small_mask = bincount < 1536
        if small_mask.any():
            small_indices = torch.where(small_mask)[0].tolist()
            small_values = bincount[small_mask].tolist()
            raise ValueError(
                f"原始样本点数过小：索引={small_indices}，点数={small_values}\n"
                "补点后最小点数应为1536（384×4），请检查补点流程"
            )
        # 原始样本必须为384的倍数（补点逻辑要求）
        if (bincount % 384 != 0).any():
            wrong_indices = torch.where(bincount % 384 != 0)[0].tolist()
            wrong_values = bincount[wrong_indices].tolist()
            raise ValueError(
                f"原始样本点数非384的倍数：索引={wrong_indices}，点数={wrong_values}\n"
                "需与补点逻辑（384倍数）保持一致"
            )
    else:
        # 下采样后样本的日志提示（非报错）
        min_points = bincount.min().item()
        if min_points < 384:
            logging.debug(f"[注意] 下采样后样本最小点数为{min_points}（小于384），offset={offset.tolist()}")

    # 4. 确保与offset同设备
    return bincount.to(offset.device)



class DummyClass:
    def __init__(self):
        pass
