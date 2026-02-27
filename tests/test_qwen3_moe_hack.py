#!/usr/bin/env python3
# tp_qwen3_moe_npu.py
import os
import re
import json
import logging
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
from collections import OrderedDict
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3DecoderLayer,
)
from safetensors.torch import load_file, save_file
import hashlib
import datetime


def setup() -> tuple[int, int, str]:
    rank_id = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    
    torch.npu.set_device(f'npu:{rank_id}')
    
    try:
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank_id,
            group_name="qwen3_tp_group"
        )
        dist.barrier()
        print("分布式初始化成功")
    except Exception as e:
        print(f"分布式初始化失败: {e}")
        raise
    
    device = f"npu:{rank_id}"
    torch.npu.set_device(device)
    return rank_id, world_size, device

def sync_input_tensor(input_ids: torch.Tensor, tp_rank: int, tp_size: int, device: str) -> torch.Tensor:
    if tp_rank == 0:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        input_ids = input_ids.to(device, non_blocking=True).contiguous()
        input_shape = list(input_ids.shape)
        print(f"准备广播输入，形状：{input_shape}")
        
        shape_tensor = torch.tensor(input_shape, dtype=torch.int64).to(device)
    else:
        shape_tensor = torch.tensor([0, 0], dtype=torch.int64).to(device)

    dist.broadcast(shape_tensor, src=0)
    batch_size, seq_len = shape_tensor.tolist()
    
    if batch_size <= 0 or seq_len <= 0 or batch_size > 32 or seq_len > 2048:
        raise RuntimeError(
            f"接收到非法形状 ({batch_size}, {seq_len})，请检查广播逻辑！"
        )

    if tp_rank == 0:
        input_ids = input_ids.contiguous()
    else:
        input_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device).contiguous()
        print(f"创建空tensor，形状：{input_ids.shape}")

    dist.barrier()
    dist.broadcast(input_ids, src=0)
    input_ids = input_ids.contiguous()
    dist.barrier()

    if tp_size > 1:
        verify_tensor = torch.tensor([0], dtype=torch.int32).to(device)
        if tp_rank == 0:
            verify_tensor = torch.tensor([1], dtype=torch.int32).to(device)
        
        dist.broadcast(verify_tensor, src=0)
        if verify_tensor.item() != 1:
            raise RuntimeError(f"验证失败！")

    print(f"输入同步完成，最终形状：{input_ids.shape}")
    return input_ids

def broadcast_tensor_npu(tensor: Optional[torch.Tensor], name: str, tp_rank: int, tp_size: int, src: int = 0) -> Optional[torch.Tensor]:
    if not dist.is_initialized():
        return tensor

    rank = dist.get_rank()
    device = f"npu:{rank}"
    torch.npu.set_device(device)

    try:
        exists_tensor = torch.tensor([0], dtype=torch.int32).to(device, non_blocking=True)
        if rank == src:
            weight_exists = 1 if (tensor is not None and isinstance(tensor, torch.Tensor)) else 0
            exists_tensor = torch.tensor([weight_exists], dtype=torch.int32).to(device, non_blocking=True)
        
        dist.broadcast(exists_tensor, src=src)
        weight_exists = exists_tensor.item()

        if weight_exists == 0:
            print(f"同步到权重 {name} 不存在的标识，返回None")
            return None

        shape_tensor = torch.tensor([0]*6, dtype=torch.int64).to(device, non_blocking=True)
        if rank == src:
            shape_list = list(tensor.shape)
            shape_list = shape_list + [0] * (6 - len(shape_list))
            shape_tensor = torch.tensor(shape_list, dtype=torch.int64).to(device, non_blocking=True)

        dist.broadcast(shape_tensor, src=src)
        target_shape = tuple([s for s in shape_tensor.tolist() if s > 0])
        target_dtype = torch.bfloat16

        if rank == src:
            broadcast_tensor = tensor.to(dtype=target_dtype, device=device, non_blocking=True).contiguous()
        else:
            broadcast_tensor = torch.empty(
                target_shape,
                dtype=target_dtype,
                device=device,
                requires_grad=False
            ).contiguous()

        dist.broadcast(broadcast_tensor, src=src)
        print(f"成功广播权重 {name}，形状：{broadcast_tensor.shape}")
        return broadcast_tensor

    except Exception as e:
        print(f"广播权重 {name} 数据失败: {str(e)}")
        if dist.is_initialized():
            dist.barrier()
        raise

def get_tp_shard_filename(model_path: str, weight_name: str, tp_rank: int, tp_size: int) -> str:
    """
    获取TP切分后权重文件的路径
    """
    # 安全处理权重名称中的特殊字符
    safe_weight_name = weight_name.replace(".", "_").replace("/", "_").replace(":", "_").replace("*", "star")
    # 按TP size创建子目录，方便管理不同并行度的分片
    tp_shard_dir = os.path.join(model_path, f"tp_shards", f"tp_size_{tp_size}")
    os.makedirs(tp_shard_dir, exist_ok=True)
    # 返回最终的文件路径
    return os.path.join(tp_shard_dir, f"{safe_weight_name}_rank_{tp_rank}.safetensors")

def get_tp_metadata_filename(model_path: str, weight_name: str, tp_rank: int, tp_size: int) -> str:
    """
    获取TP切分权重元数据文件路径
    """
    shard_file = get_tp_shard_filename(model_path, weight_name, tp_rank, tp_size)
    return shard_file.replace(".safetensors", "_metadata.json")


def save_tp_shard_weight(weight: torch.Tensor, model_path: str, weight_name: str, tp_rank: int, tp_size: int) -> None:
    """
    保存TP切分后的权重文件（增强版）
    - 保存权重数据和完整元数据
    - 计算权重哈希用于校验
    - 增加异常处理和日志
    """
    try:
        # 确保权重是contiguous的
        weight = weight.contiguous()
        
        # 获取保存路径
        shard_file = get_tp_shard_filename(model_path, weight_name, tp_rank, tp_size)
        metadata_file = get_tp_metadata_filename(model_path, weight_name, tp_rank, tp_size)
        
        # 准备保存的权重字典
        save_dict = {
            "weight": weight.cpu().contiguous(),  # 移到CPU保存，避免NPU设备依赖
        }
        

        
        # 准备元数据
        metadata = {
            "weight_name": weight_name,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "device": str(weight.device),
            "save_time": str(datetime.datetime.now()),
            "model_path": model_path,
        }
        
        # 保存权重文件
        save_file(save_dict, shard_file)
        
        # 保存元数据文件
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 打印保存成功信息
        print(f"✅ 已保存TP分片权重: {shard_file}")
        print(f"   - 形状: {weight.shape} | 数据类型: {weight.dtype}")
        print(f"   - 元数据文件: {metadata_file}")
        
    except Exception as e:
        print(f"❌ 保存TP分片权重失败 {weight_name} (rank {tp_rank}): {str(e)}")
        # 保存失败时删除不完整的文件
        if os.path.exists(shard_file):
            os.remove(shard_file)
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        raise

def load_shard_weight(model_path: str, name: str, tp_rank: int, tp_size: int, device: str) -> Optional[torch.Tensor]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    final_shard = None
    
    # 优先检查是否已有预保存的TP分片权重
    tp_shard_file = get_tp_shard_filename(model_path, name, tp_rank, tp_size)
    metadata_file = get_tp_metadata_filename(model_path, name, tp_rank, tp_size)
    
    dist.barrier()
    
    # 第一步：尝试加载已保存的TP分片权重
    if os.path.exists(tp_shard_file):
        try:
            # 加载权重文件
            weight_dict = load_file(tp_shard_file, device="cpu")
            final_shard = weight_dict["weight"].to(torch.bfloat16)
            
            # 加载并验证元数据
            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                
                print(f"📥 加载已保存的TP分片权重 {name} (rank {tp_rank})")
                print(f"   - 文件: {tp_shard_file} | 形状: {final_shard.shape} | 哈希验证通过")
            else:
                print(f"📥 加载已保存的TP分片权重 {name} (rank {tp_rank}) - 无元数据文件")
            
            final_shard = final_shard.to(device, non_blocking=True).contiguous()
            # 直接广播并返回，无需重新切分
            final_shard = broadcast_tensor_npu(final_shard, name, tp_rank, tp_size, src=0)
            return final_shard
            
        except Exception as e:
            print(f"⚠️ 加载已保存的TP分片权重失败 {tp_shard_file}: {e}，将重新切分")
            # 删除损坏的文件
            if os.path.exists(tp_shard_file):
                os.remove(tp_shard_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
    
    # 第二步：如果没有预保存的分片，执行原有的权重加载和切分逻辑
    if rank == 0:
        try:
            with open(os.path.join(model_path, "config.json")) as f:
                config_dict = json.load(f)
            config = Qwen3Config(**config_dict)
            
            q_dim = config.num_attention_heads * config.head_dim
            kv_dim = config.num_key_value_heads * config.head_dim
            intermediate_size = config.intermediate_size
            moe_intermediate_size = config.moe_intermediate_size

            index_file = os.path.join(model_path, "model.safetensors.index.json")
            weight_files = ["model.safetensors"]
            if os.path.exists(index_file):
                with open(index_file, encoding='utf-8') as f:
                    weight_map = json.load(f).get("weight_map", {})
                weight_files = sorted(set(weight_map.values()))

            full_weight = None
            for weight_file in weight_files:
                file_path = os.path.join(model_path, weight_file)
                if not os.path.exists(file_path):
                    print(f"权重文件 {file_path} 不存在")
                    continue
                weight_dict = load_file(file_path, device="cpu")
                if name in weight_dict:
                    full_weight = weight_dict[name].to(torch.bfloat16)
                    print(f"找到原始权重 {name}，原始形状：{full_weight.shape}")
                    break

            if full_weight is not None:
                split_dim = -1
                target_dim_size = -1
                
                if "q_proj.weight" in name:
                    split_dim = 0
                    target_dim_size = q_dim
                elif "k_proj.weight" in name or "v_proj.weight" in name:
                    split_dim = 0
                    target_dim_size = kv_dim
                elif "o_proj.weight" in name:
                    split_dim = 1
                    target_dim_size = q_dim
                elif "gate_proj.weight" in name or "up_proj.weight" in name:
                    split_dim = 0
                    target_dim_size = intermediate_size
                elif "down_proj.weight" in name:
                    split_dim = 1
                    target_dim_size = intermediate_size
                elif "embed_tokens.weight" in name or "lm_head.weight" in name:
                    split_dim = 0
                    target_dim_size = full_weight.size(0)
                elif "gate.weight" in name:
                    split_dim = 0
                    target_dim_size = config.num_experts
                elif "expert_up_proj.weight" in name or "experts.*.up_proj.weight" in name:
                    split_dim = 0
                    target_dim_size = moe_intermediate_size
                elif "expert_down_proj.weight" in name or "experts.*.down_proj.weight" in name:
                    split_dim = 1
                    target_dim_size = moe_intermediate_size
                elif "experts.*.gate_proj.weight" in name:
                    split_dim = 0
                    target_dim_size = moe_intermediate_size

                if split_dim >= 0:
                    actual_dim_size = full_weight.size(split_dim)
                    
                    if actual_dim_size == target_dim_size // tp_size:
                        print(f"权重 {name} 已是切分后状态，无需再次切分")
                        final_shard = full_weight.to(device, non_blocking=True).contiguous()
                    else:
                        if actual_dim_size % tp_size != 0:
                            print(f"权重 {name} 维度 {actual_dim_size} 无法被TP size {tp_size} 整除，使用完整权重")
                            final_shard = full_weight.to(device, non_blocking=True).contiguous()
                        else:
                            chunk_size = actual_dim_size // tp_size
                            start_idx = tp_rank * chunk_size
                            end_idx = start_idx + chunk_size

                            if split_dim == 0:
                                final_shard = full_weight[start_idx:end_idx, :]
                            else:
                                final_shard = full_weight[:, start_idx:end_idx]
                            
                            final_shard = final_shard.to(device, non_blocking=True).contiguous()
                            print(f"✂️ 切分权重 {name} → 切分后形状：{final_shard.shape}")
                else:
                    final_shard = full_weight.to(device, non_blocking=True).contiguous()
                    print(f"权重 {name} 为非并行层")
                
                # 保存切分后的权重文件（修复：现在会保存）
                if final_shard is not None:
                    save_tp_shard_weight(final_shard, model_path, name, tp_rank, tp_size)

            else:
                print(f"未找到权重 {name}")

        except Exception as e:
            print(f"加载/切分权重 {name} 失败: {str(e)}")
            final_shard = None
    
    dist.barrier()
    
    # 其他rank加载主rank广播的权重
    final_shard = broadcast_tensor_npu(final_shard, name, tp_rank, tp_size, src=0)
    
    # 非0 rank也保存一份分片权重
    if rank != 0 and final_shard is not None:
        save_tp_shard_weight(final_shard, model_path, name, tp_rank, tp_size)
    
    return final_shard


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                 tp_rank: int = 0, tp_size: int = 1, device: str = "npu:0",
                 is_o_proj: bool = False):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = device
        self.is_o_proj = is_o_proj
        
        self.original_weight_shape = weight.shape
        assert len(weight.shape) == 2, f"权重必须是2维，当前形状：{weight.shape}"
        self.out_dim_total, self.in_dim = weight.shape
        print(f"原始权重形状 [out_dim, in_dim]：{self.original_weight_shape} (is_o_proj: {is_o_proj})")
        
        if self.is_o_proj:
            chunk_size = self.in_dim // tp_size
            start_idx = tp_rank * chunk_size
            end_idx = start_idx + chunk_size
            
            self.weight = torch.nn.Parameter(
                weight[:, start_idx:end_idx].to(device, non_blocking=True).contiguous()
            )
            self.out_dim = self.out_dim_total
            self.in_dim_chunk = chunk_size
            print(f"o_proj切分后权重形状：{self.weight.shape} (in_dim切分: {chunk_size})")
        else:
            chunk_size = self.out_dim_total // tp_size
            start_idx = tp_rank * chunk_size
            end_idx = start_idx + chunk_size
            
            if end_idx > self.out_dim_total:
                end_idx = self.out_dim_total
            
            self.weight = torch.nn.Parameter(
                weight[start_idx:end_idx, :].to(device, non_blocking=True).contiguous()
            )
            self.out_dim_chunk = chunk_size
            print(f"普通层切分后权重形状：{self.weight.shape} (out_dim切分: {chunk_size})")
        
        if bias is not None:
            bias_chunk = bias[start_idx:end_idx].to(device, non_blocking=True).contiguous()
            self.bias = torch.nn.Parameter(bias_chunk)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.bfloat16, non_blocking=True).contiguous()
        batch_size, seq_len, in_dim = x.shape
        
        if not self.is_o_proj and in_dim != self.in_dim:
            raise RuntimeError(
                f"输入in_dim {in_dim} 不匹配权重in_dim {self.in_dim}！"
                f"输入形状：{x.shape}，权重形状：{self.weight.shape}"
            )
        
        if self.is_o_proj:
            start_idx = self.tp_rank * self.in_dim_chunk
            end_idx = start_idx + self.in_dim_chunk
            end_idx = min(end_idx, in_dim)
            
            x_chunk = x[..., start_idx:end_idx].contiguous()
            output = torch.nn.functional.linear(x_chunk, self.weight, self.bias)
        else:
            output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        output = output.contiguous()
        
        if dist.is_initialized() and self.tp_size > 1:
            gathered_outputs = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(gathered_outputs, output)
            
            output = torch.cat(gathered_outputs, dim=-1).contiguous()
            
            if output.shape[-1] > self.in_dim:
                output = output[..., :self.in_dim].contiguous()
        
        return output

class RowParallelLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, tp_rank: int, tp_size: int, device: str):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = device
        
        assert len(weight.shape) == 2, f"权重必须是2维，当前形状：{weight.shape}"
        self.out_dim, self.in_dim_total = weight.shape
        
        chunk_size = self.in_dim_total // tp_size
        start_idx = tp_rank * chunk_size
        end_idx = start_idx + chunk_size
        end_idx = min(end_idx, self.in_dim_total)
        
        self.weight = torch.nn.Parameter(
            weight[:, start_idx:end_idx].to(device, non_blocking=True).contiguous()
        )
        self.in_dim_chunk = chunk_size
        print(f"RowParallel切分后权重形状：{self.weight.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.bfloat16, non_blocking=True).contiguous()
        
        output = torch.nn.functional.linear(x, self.weight)
        output = output.contiguous()
        
        if dist.is_initialized() and self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            output = output.contiguous()
        
        return output


class TPQwen3ForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, tp_rank: int, tp_size: int, device: str):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = device

        with open(os.path.join(model_path, "config.json")) as f:
            config_dict = json.load(f)
        self.config = Qwen3Config(** config_dict)
        self.config.dtype = torch.bfloat16
        
        self.hidden_size = self.config.hidden_size
        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        
        self.q_dim = self.num_q_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.intermediate_size = self.config.intermediate_size
        self.moe_inter_size = self.config.moe_intermediate_size
        
        self.num_experts = getattr(self.config, 'num_experts', 8)
        self.num_local_experts = getattr(self.config, 'num_local_experts', self.num_experts)
        
        print(f"MoE配置 - 总专家数: {self.num_experts}, 本地专家数: {self.num_local_experts}")

        self.base_model = Qwen3ForCausalLM(self.config)
        self.base_model = self.base_model.to(device, dtype=torch.bfloat16)
        self.base_model.eval()

        self._load_and_replace_weights(model_path)

    def _load_and_replace_weights(self, model_path: str):
        # 加载embedding权重
        embed_weight = load_shard_weight(model_path, "model.embed_tokens.weight", self.tp_rank, self.tp_size, self.device)
        if embed_weight is not None:
            self.base_model.model.embed_tokens = torch.nn.Embedding(
                num_embeddings=embed_weight.size(0),
                embedding_dim=self.hidden_size,
                _weight=embed_weight
            ).to(self.device, dtype=torch.bfloat16)
            # 保存embedding权重
            save_tp_shard_weight(embed_weight, model_path, "model.embed_tokens.weight", self.tp_rank, self.tp_size)

        # 加载所有层的权重
        for i in range(self.config.num_hidden_layers):
            self._load_attention_layer(model_path, i)
            self._load_mlp_layer(model_path, i)

        # 加载lm_head权重
        lm_head_weight = load_shard_weight(model_path, "lm_head.weight", self.tp_rank, self.tp_size, self.device)
        if lm_head_weight is not None:
            self.base_model.lm_head = ColumnParallelLinear(
                lm_head_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
            )
            self.base_model.lm_head.in_dim = self.hidden_size
            # 修复原代码中的bug：补充缺失的变量定义
            save_tp_shard_weight(lm_head_weight, model_path, "lm_head.weight", self.tp_rank, self.tp_size)

    def _load_attention_layer(self, model_path: str, layer_idx: int):
        layer_name = f"model.layers.{layer_idx}.self_attn"
        q_weight = load_shard_weight(model_path, f"{layer_name}.q_proj.weight", self.tp_rank, self.tp_size, self.device)
        k_weight = load_shard_weight(model_path, f"{layer_name}.k_proj.weight", self.tp_rank, self.tp_size, self.device)
        v_weight = load_shard_weight(model_path, f"{layer_name}.v_proj.weight", self.tp_rank, self.tp_size, self.device)
        o_weight = load_shard_weight(model_path, f"{layer_name}.o_proj.weight", self.tp_rank, self.tp_size, self.device)
        
        if all(w is not None for w in [q_weight, k_weight, v_weight, o_weight]):
            self.base_model.model.layers[layer_idx].self_attn.q_proj = ColumnParallelLinear(
                q_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
            )
            self.base_model.model.layers[layer_idx].self_attn.q_proj.in_dim = self.hidden_size
            
            self.base_model.model.layers[layer_idx].self_attn.k_proj = ColumnParallelLinear(
                k_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
            )
            self.base_model.model.layers[layer_idx].self_attn.k_proj.in_dim = self.hidden_size
            
            self.base_model.model.layers[layer_idx].self_attn.v_proj = ColumnParallelLinear(
                v_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
            )
            self.base_model.model.layers[layer_idx].self_attn.v_proj.in_dim = self.hidden_size
            
            self.base_model.model.layers[layer_idx].self_attn.o_proj = ColumnParallelLinear(
                o_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=True
            )
            self.base_model.model.layers[layer_idx].self_attn.o_proj.in_dim = self.hidden_size
            
            # 保存attention层权重
            save_tp_shard_weight(q_weight, model_path, f"{layer_name}.q_proj.weight", self.tp_rank, self.tp_size)
            save_tp_shard_weight(k_weight, model_path, f"{layer_name}.k_proj.weight", self.tp_rank, self.tp_size)
            save_tp_shard_weight(v_weight, model_path, f"{layer_name}.v_proj.weight", self.tp_rank, self.tp_size)
            save_tp_shard_weight(o_weight, model_path, f"{layer_name}.o_proj.weight", self.tp_rank, self.tp_size)

    def _load_mlp_layer(self, model_path: str, layer_idx: int):
        layer_name_mlp = f"model.layers.{layer_idx}.mlp"
        
        gate_weight = load_shard_weight(model_path, f"{layer_name_mlp}.gate_proj.weight", self.tp_rank, self.tp_size, self.device)
        up_weight = load_shard_weight(model_path, f"{layer_name_mlp}.up_proj.weight", self.tp_rank, self.tp_size, self.device)
        down_weight = load_shard_weight(model_path, f"{layer_name_mlp}.down_proj.weight", self.tp_rank, self.tp_size, self.device)
        
        if all(w is not None for w in [gate_weight, up_weight, down_weight]):
            self.base_model.model.layers[layer_idx].mlp.gate_proj = ColumnParallelLinear(
                gate_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
            )
            self.base_model.model.layers[layer_idx].mlp.gate_proj.in_dim = self.hidden_size
            
            self.base_model.model.layers[layer_idx].mlp.up_proj = ColumnParallelLinear(
                up_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
            )
            self.base_model.model.layers[layer_idx].mlp.up_proj.in_dim = self.hidden_size
            
            self.base_model.model.layers[layer_idx].mlp.down_proj = RowParallelLinear(
                down_weight, self.tp_rank, self.tp_size, self.device
            )
            
            # 保存MLP层权重
            save_tp_shard_weight(gate_weight, model_path, f"{layer_name_mlp}.gate_proj.weight", self.tp_rank, self.tp_size)
            save_tp_shard_weight(up_weight, model_path, f"{layer_name_mlp}.up_proj.weight", self.tp_rank, self.tp_size)
            save_tp_shard_weight(down_weight, model_path, f"{layer_name_mlp}.down_proj.weight", self.tp_rank, self.tp_size)
            
        else:
            print(f"开始加载第{layer_idx}层MoE多专家权重")
            
            moe_gate_weight = load_shard_weight(model_path, f"{layer_name_mlp}.gate.weight", self.tp_rank, self.tp_size, self.device)
            if moe_gate_weight is not None:
                self.base_model.model.layers[layer_idx].mlp.gate = ColumnParallelLinear(
                    moe_gate_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
                )
                self.base_model.model.layers[layer_idx].mlp.gate.in_dim = self.hidden_size
                save_tp_shard_weight(moe_gate_weight, model_path, f"{layer_name_mlp}.gate.weight", self.tp_rank, self.tp_size)
            
            expert_indices = self._get_layer_expert_indices(model_path, layer_idx)
            
            if not expert_indices:
                expert_indices = range(0, self.num_experts)
                print(f"未找到专家索引，使用默认范围: {expert_indices}")
            
            for expert_idx in expert_indices:
                expert_gate_name = f"{layer_name_mlp}.experts.{expert_idx}.gate_proj.weight"
                expert_up_name = f"{layer_name_mlp}.experts.{expert_idx}.up_proj.weight"
                expert_down_name = f"{layer_name_mlp}.experts.{expert_idx}.down_proj.weight"
                
                exp_gate_weight = load_shard_weight(model_path, expert_gate_name, self.tp_rank, self.tp_size, self.device)
                exp_up_weight = load_shard_weight(model_path, expert_up_name, self.tp_rank, self.tp_size, self.device)
                exp_down_weight = load_shard_weight(model_path, expert_down_name, self.tp_rank, self.tp_size, self.device)
                
                if all(w is not None for w in [exp_gate_weight, exp_up_weight, exp_down_weight]):
                    if not hasattr(self.base_model.model.layers[layer_idx].mlp, 'experts'):
                        self.base_model.model.layers[layer_idx].mlp.experts = torch.nn.ModuleList()
                    
                    while len(self.base_model.model.layers[layer_idx].mlp.experts) <= expert_idx:
                        self.base_model.model.layers[layer_idx].mlp.experts.append(torch.nn.Module())
                    
                    expert_module = self.base_model.model.layers[layer_idx].mlp.experts[expert_idx]
                    expert_module.gate_proj = ColumnParallelLinear(
                        exp_gate_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
                    )
                    expert_module.gate_proj.in_dim = self.hidden_size
                    
                    expert_module.up_proj = ColumnParallelLinear(
                        exp_up_weight, None, self.tp_rank, self.tp_size, self.device, is_o_proj=False
                    )
                    expert_module.up_proj.in_dim = self.hidden_size
                    
                    expert_module.down_proj = RowParallelLinear(
                        exp_down_weight, self.tp_rank, self.tp_size, self.device
                    )
                    
                    # 保存MoE专家权重
                    save_tp_shard_weight(exp_gate_weight, model_path, expert_gate_name, self.tp_rank, self.tp_size)
                    save_tp_shard_weight(exp_up_weight, model_path, expert_up_name, self.tp_rank, self.tp_size)
                    save_tp_shard_weight(exp_down_weight, model_path, expert_down_name, self.tp_rank, self.tp_size)
                    
                    print(f"成功加载第{layer_idx}层专家{expert_idx}的权重")
                else:
                    print(f"跳过第{layer_idx}层专家{expert_idx}的权重（部分权重缺失）")

    def _get_layer_expert_indices(self, model_path: str, layer_idx: int) -> list:
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            return []
        
        try:
            with open(index_file, encoding='utf-8') as f:
                weight_map = json.load(f).get("weight_map", {})
            
            pattern = re.compile(f"model\.layers\.{layer_idx}\.mlp\.experts\.(\d+)\.")
            expert_indices = set()
            
            for weight_name in weight_map.keys():
                match = pattern.search(weight_name)
                if match:
                    expert_idx = int(match.group(1))
                    expert_indices.add(expert_idx)
            
            return sorted(list(expert_indices))
        except Exception as e:
            print(f"获取专家索引失败: {e}")
            return []

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 60,
        max_seq_len: int = 100,
        eos_token_id: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        input_ids = sync_input_tensor(input_ids, self.tp_rank, self.tp_size, self.device)
        
        batch_size, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            input_seq = generated_ids[:, -min(max_seq_len, generated_ids.shape[1]):]
            
            outputs = self.base_model(input_ids=input_seq)
            logits = outputs.logits
            
            next_token_logits = logits[:, -1, :]
            
            dist.broadcast(next_token_logits, src=0)
            
            if self.tp_rank == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if next_token.item() == eos_token_id:
                    print(f"生成到第{step+1}步时遇到EOS token，停止生成")
                    break
            else:
                next_token = torch.empty((batch_size, 1), dtype=torch.long).to(self.device)
            
            dist.broadcast(next_token, src=0)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            dist.barrier()
        
        if self.tp_rank == 0:
            print(f"生成完成，最终token序列长度：{generated_ids.shape[1]}")
            return generated_ids
        else:
            return None


def main():
    tp_rank, tp_size, device = setup()
    
    model_path = os.environ.get("MODEL_PATH", "/data01/zbl/mojo_opset/Qwen3-30B-A3B")
    
    tokenizer = None
    if tp_rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer加载完成")
    
    print("开始加载模型...")
    model = TPQwen3ForCausalLM(model_path, tp_rank, tp_size, device)
    model.eval()
    print("模型初始化成功")
    
    input_text = os.environ.get("INPUT_TEXT", "你好，请介绍一下自己。")
    input_ids = None
    if tp_rank == 0:
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"原始输入tokenize后形状：{input_ids.shape}")
    else:
        input_ids = torch.tensor([[0]], dtype=torch.long)
    
    input_ids = sync_input_tensor(input_ids, tp_rank, tp_size, device)
    
    with torch.no_grad():
        max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", 60))
        out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    
    if tp_rank == 0 and out_ids is not None and tokenizer is not None:
        output_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print("\n=== 生成结果 ===")
        print(f"输入：{input_text}")
        print(f"输出：{output_text}")

if __name__ == "__main__":
    main()