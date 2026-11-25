# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

from PIL import Image
from ..utils.dataset import collate_fn
from .kl_prcp_utils import random_patch_blackening

from tensordict import TensorDict

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics

def apply_kl_penalty_caption(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    scores = data.batch["token_level_scores"]            # [B, L]
    mask   = data.batch["response_mask"]                 # [B, L] 形如 [1,1,1,0,0,0]
    B, L   = mask.shape

    # 逐 token KL（只做统计，不反传）
    kld_tok = compute_kl(
        data.batch["old_log_probs"].detach(),
        data.batch["aug_log_probs_caption"].detach(),
        kl_penalty=kl_penalty,
    ).detach()                                           # [B, L]
    # 如果 kl_penalty="kl" 可能出现负值，想严格“KL≥0”可 relu 一下
    # kld_tok = torch.relu(kld_tok)

    # 每序列 KL 均值（只在有效位置统计） -> [B]
    seq_mean_kl = VF.masked_mean(kld_tok, mask, dim=-1)

    # 找“最后一个有效 token”的 one-hot：由于是 [1,1,1,0,0]，最后一个有效就是 sum-1
    valid_len   = mask.sum(dim=1)                        # [B]
    last_idx    = (valid_len.clamp_min(1) - 1).long()    # [B] 防 0 长
    last_onehot = torch.nn.functional.one_hot(last_idx, num_classes=L).to(mask.dtype).to(mask.device)  # [B, L]

    # 只在最后一个有效 token 处扣 mean-KL 惩罚（长度无关）
    coef    = kl_ctrl.kl_coef
    penalty = (coef * seq_mean_kl).unsqueeze(1) * last_onehot
    data.batch["token_level_rewards"] = scores - penalty

    # 日志与调度器更新（用 mean-KL 的 batch 均值）
    current_kl = seq_mean_kl.mean().item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": coef}
    kl_ctrl.update(current_kl=current_kl, n_steps=B)
    return data, metrics


# def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
#     token_level_rewards = data.batch["token_level_rewards"]
#     response_mask = data.batch["response_mask"]
#     index = data.non_tensor_batch["uid"]
#     if adv_estimator == AdvantageEstimator.GAE:
#         values = data.batch["values"]
#         advantages, returns = core_algos.compute_gae_advantage_return(
#             token_level_rewards, values, response_mask, gamma, lam
#         )
#     elif adv_estimator == AdvantageEstimator.GRPO:
#         advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
#     elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
#         advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
#             token_level_rewards, response_mask, gamma
#         )
#     elif adv_estimator == AdvantageEstimator.REMAX:
#         reward_baselines = data.batch["reward_baselines"]
#         advantages, returns = core_algos.compute_remax_outcome_advantage(
#             token_level_rewards, reward_baselines, response_mask
#         )
#     elif adv_estimator == AdvantageEstimator.RLOO:
#         advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
#     else:
#         raise NotImplementedError

#     data.batch["advantages"] = advantages
#     data.batch["returns"] = returns
#     return data

def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    *,
    norm_mode: str = "minmax_symmetric",   # "minmax_symmetric" 或 "zscore"
    beta: float = 1.0,                     # 指数权重强度
    w_min: float = 0.5,                    # 权重下界
    w_max: float = 1.5,                    # 权重上界
    z_max: float = 3.0,                    # zscore 截断阈值 |z|<=z_max
):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]

    # —— 用于分组的 id：优先 uid_origin（同一 prompt 的多个采样同组），否则 uid ——
    index = data.non_tensor_batch.get("uid_origin", data.non_tensor_batch["uid"])

    # 1) 先按原算法算出 advantages / returns
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    # 2) 计算每样本 mean-KL（只在 mask=1 的 token 上）
    old_lp = data.batch.get("old_log_probs", None)
    if old_lp is None:
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    # KL 参考优先级：aug_log_probs_caption > aug_log_probs > ref_log_probs（如需仅 caption，可保留当前写法）
    if "aug_log_probs_caption" in data.batch:
        ref_lp = data.batch["aug_log_probs_caption"]
        kl_penalty = "abs"
    elif "aug_log_probs" in data.batch:
        ref_lp = data.batch["aug_log_probs"]
        kl_penalty = "abs"
    elif "ref_log_probs" in data.batch:
        ref_lp = data.batch["ref_log_probs"]
        kl_penalty = "abs"
    else:
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    with torch.no_grad():
        kld_tok = compute_kl(old_lp.detach(), ref_lp.detach(), kl_penalty)  # [B, L]
        seq_mean_kl = VF.masked_mean(kld_tok, response_mask, dim=-1)        # [B]

    # 2.5) 计算“序列级 advantage 的符号”，用于分流权重（按样本同一权重，便于稳定）
    with torch.no_grad():
        if advantages.dim() == 2:
            seq_mean_adv = VF.masked_mean(advantages, response_mask, dim=-1)  # [B]
        elif advantages.dim() == 1:
            seq_mean_adv = advantages
        else:
            # 非常规形状，直接不做权重
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            return data
        is_pos = (seq_mean_adv >= 0.0)  # [B] True=正优势, False=负优势

    # 3) 组内归一化（到 [-1,1] 或 z-score），并做“正负分流”的指数权重
    import numpy as _np
    from collections import defaultdict
    EPS = 1e-8
    uids = _np.array(index, dtype=object)
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)

    adv_weights = torch.ones_like(seq_mean_kl)

    for uid, idxs in groups.items():
        idxs_t = torch.as_tensor(idxs, device=seq_mean_kl.device, dtype=torch.long)
        kl_g   = seq_mean_kl.index_select(0, idxs_t)         # [G]
        pos_g  = is_pos.index_select(0, idxs_t)              # [G], bool

        if kl_g.numel() == 1:
            kl_norm = torch.zeros_like(kl_g)                 # 单样本组 → 归一化为 0
        else:
            if norm_mode == "minmax_symmetric":
                kmin, kmax = kl_g.min(), kl_g.max()
                denom  = (kmax - kmin).clamp_min(EPS)
                kl01   = (kl_g - kmin) / denom
                kl_norm = 2.0 * kl01 - 1.0                   # [-1, 1]
            elif norm_mode == "zscore":
                mu = kl_g.mean()
                std = kl_g.std(unbiased=False).clamp_min(EPS)
                kl_norm = (kl_g - mu) / std
                if z_max is not None and z_max > 0:
                    kl_norm = kl_norm.clamp(-z_max, z_max)
            else:
                raise ValueError(f"Unknown norm_mode: {norm_mode}")

        # 正负分流：正优势用 exp(-beta * kl_norm)，负优势用 exp(+beta * kl_norm)
        w_pos = torch.exp(-beta * kl_norm)
        w_neg = torch.exp(+beta * kl_norm)
        w_raw = torch.where(pos_g, w_pos, w_neg)

        # 组内均值归一（均值=1）
        # w = w_raw / w_raw.mean().clamp_min(EPS)
        w = w_raw
        w = w.clamp(w_min, w_max)

        adv_weights.index_copy_(0, idxs_t, w)

        # ===== 🔍 打印调试信息 =====
        print(f"[Adv KL Group={uid}]")
        print(f"  raw KL:    {kl_g.detach().cpu().numpy()}")
        print(f"  norm KL:   {kl_norm.detach().cpu().numpy()}")
        print(f"  seq_adv+:  {pos_g.detach().cpu().numpy()}  (True=pos, False=neg)")
        print(f"  w_final:   {w.detach().cpu().numpy()}")

        # 打印 advantage 前后 —— 逐样本，而不是乘 w.mean()
        if advantages.dim() == 2:
            adv_seq_before = VF.masked_mean(
                advantages.index_select(0, idxs_t),
                response_mask.index_select(0, idxs_t),
                dim=-1,                 # [G]
            )
            adv_seq_after = adv_seq_before * w   # 逐样本缩放
        else:
            adv_seq_before = advantages.index_select(0, idxs_t)  # [G]
            adv_seq_after  = adv_seq_before * w

        print(f"  adv_before: {adv_seq_before.detach().cpu().numpy()}")
        print(f"  adv_after : {adv_seq_after.detach().cpu().numpy()}")
        print(f"  delta     : {(adv_seq_after - adv_seq_before).detach().cpu().numpy()}")

    # 4) 用组内权重缩放 advantages（兼容 token-level / outcome-level）
    if advantages.dim() == 2:
        advantages = advantages * adv_weights.unsqueeze(1)   # token-level
    elif advantages.dim() == 1:
        advantages = advantages * adv_weights                # outcome-level

    # 5) 回填（附带中间量便于监控）
    data.batch["advantages"] = advantages
    data.batch["returns"]    = returns
    data.batch["adv_weights_kl_group"] = adv_weights
    data.batch["seq_mean_kl_for_adv"]  = seq_mean_kl
    return data


def merge_batch_with_caption_batch(batch, aug_caption_batch):
    """
    将 batch 和 aug_caption_batch 进行融合（concatenate），
    自动对齐字段，补齐缺失字段，确保 TensorDict 和非tensor字段一致性。
    """

    # ========= 🔹 1. TensorDict（tensor batch）字段拼接 =========
    merged_tensor_dict = {}

    # 所有字段名的并集
    all_keys = set(batch.batch.keys()).union(set(aug_caption_batch.batch.keys()))

    for key in all_keys:
        t1 = batch.batch.get(key, None)
        t2 = aug_caption_batch.batch.get(key, None)

        if t1 is None and t2 is not None:
            t1 = t2.clone()
        elif t2 is None and t1 is not None:
            t2 = t1.clone()
        elif t1 is None and t2 is None:
            raise ValueError(f"[merge_batch_with_caption_batch] Key '{key}' missing in both batches.")

        merged_tensor_dict[key] = torch.cat([t1, t2], dim=0)

    batch.batch = TensorDict(merged_tensor_dict, batch_size=[merged_tensor_dict["input_ids"].shape[0]])


    # ========= 🔸 2. 非tensor字段拼接（如 string prompt） =========
    merged_non_tensor_dict = {}
    input_ids_1 = batch.batch["input_ids"]
    input_ids_2 = aug_caption_batch.batch["input_ids"]
    bs1 = input_ids_1.shape[0]
    bs2 = input_ids_2.shape[0]

    all_non_tensor_keys = set(batch.non_tensor_batch.keys()).union(set(aug_caption_batch.non_tensor_batch.keys()))

    for key in all_non_tensor_keys:
        l1 = batch.non_tensor_batch.get(key, None)
        l2 = aug_caption_batch.non_tensor_batch.get(key, None)

        # 如果两个都缺失，报错
        if l1 is None and l2 is None:
            raise ValueError(f"[merge_batch_with_caption_batch] Key '{key}' is missing in both batches.")

        # 如果一方缺失，就复制另一方的数据
        if l1 is None and l2 is not None:
            l1 = l2.copy()
        elif l2 is None and l1 is not None:
            l2 = l1.copy()

        # 如果是 multi_modal_inputs，补 None
        if key == "multi_modal_inputs":
            if not isinstance(l1, list):
                l1 = [None] * bs1
            elif len(l1) != bs1:
                l1 = [None] * bs1

            if not isinstance(l2, list):
                l2 = [None] * bs2
            elif len(l2) != bs2:
                l2 = [None] * bs2

        else:
            # 非 multi_modal_inputs 普通字段，确保为 list
            if not isinstance(l1, list):
                l1 = list(l1)
            if not isinstance(l2, list):
                l2 = list(l2)

        merged_non_tensor_dict[key] = l1 + l2

    batch.non_tensor_batch = merged_non_tensor_dict

    # ========= 🔸 3. meta_info 合并（最常用字段是 global_token_num） =========
    for key in set(batch.meta_info.keys()).union(set(aug_caption_batch.meta_info.keys())):
        v1 = batch.meta_info.get(key, [])
        v2 = aug_caption_batch.meta_info.get(key, [])

        if isinstance(v1, list) and isinstance(v2, list):
            batch.meta_info[key] = v1 + v2
        else:
            # 默认覆盖或保留原始 batch 的字段
            batch.meta_info[key] = batch.meta_info.get(key, v2)

    return batch

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        self.kl_ctrl_reward = get_kl_controller(config.algorithm)
        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        # define KL schedular
        if config.algorithm.use_kl_prcp:
            if config.algorithm.kl_prcp_schedule == "fixed":
                self.kl_ctrl_contrastive = core_algos.FixedKLController(init_kl_coef=config.algorithm.kl_prcp_coef)
            elif config.algorithm.kl_prcp_schedule == "annealing":
                start_value = config.algorithm.kl_prcp_schedule_args.get("start_value", config.algorithm.kl_prcp_coef)
                end_value = config.algorithm.kl_prcp_schedule_args.get("end_value", 0.0)
                annealing_ratio = config.algorithm.kl_prcp_schedule_args.get("annealing_ratio", 0.5)
                annealing_steps = int(self.training_steps * annealing_ratio)
                print(f"Using annealing KL schedule with start value {start_value}, end value {end_value}, and total steps {annealing_steps}.")
                self.kl_ctrl_contrastive = core_algos.AnnealingKLController(
                    start_value=start_value,
                    end_value=end_value,
                    total_steps=annealing_steps,
                )

        # define KL schedular caption
        if config.algorithm.use_kl_prcp_caption:
            if config.algorithm.kl_prcp_schedule == "fixed":
                self.kl_ctrl_contrastive_caption = core_algos.FixedKLController(init_kl_coef=config.algorithm.kl_prcp_coef)
            # elif config.algorithm.kl_prcp_schedule == "annealing":
            #     start_value = config.algorithm.kl_prcp_schedule_args.get("start_value", config.algorithm.kl_prcp_coef)
            #     end_value = config.algorithm.kl_prcp_schedule_args.get("end_value", 0.0)
            #     annealing_ratio = config.algorithm.kl_prcp_schedule_args.get("annealing_ratio", 0.5)
            #     annealing_steps = int(self.training_steps * annealing_ratio)
            #     print(f"Using annealing KL schedule with start value {start_value}, end value {end_value}, and total steps {annealing_steps}.")
            #     self.kl_ctrl_contrastive_caption = core_algos.AnnealingKLController(
            #         start_value=start_value,
            #         end_value=end_value,
            #         total_steps=annealing_steps,
            #     )


        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _aug_img_for_kl_prcp(self, original_images_pil: List[Image.Image]) -> List[Image.Image]:
        """
        Perform augmentation on the original images for contrastive KL.
        This function should be implemented based on the specific augmentation method used.
        """
        aug_config = self.config.algorithm.aug_config
        if self.config.algorithm.contrastive_type == "augmented":
            augmented_images = []
            for img in original_images_pil:
                aug_img = random_patch_blackening(img, **aug_config)
                augmented_images.append(aug_img)
            return augmented_images
        else:
            raise NotImplementedError(f"Unknown contrastive KL type: {self.config.algorithm.contrastive_type}.")

    def _get_kl_prcp_weights(self, batch: DataProto, reward_metrics: Dict[str, Any]) -> DataProto:
        if self.config.algorithm.kl_prcp_apply_mode == "all":
            batch.non_tensor_batch["kl_prcp_weighting"] = np.array([1.0] * len(batch.batch))
        elif self.config.algorithm.kl_prcp_apply_mode == "correct_only":
            weights = []
            for i in range(len(batch.batch)):
                if reward_metrics['accuracy'][i] > 0.1:
                    weights.append(1.0)
                else:
                    weights.append(0.0)
            batch.non_tensor_batch["kl_prcp_weighting"] = np.array(weights)
        elif self.config.algorithm.kl_prcp_apply_mode == "acc_weighted":
            weights = []
            for i in range(len(batch.batch)):
                if reward_metrics['accuracy'][i] > 0.1:
                    weights.append(1.0)
                else:
                    weights.append(self.config.algorithm.incorrect_weighting)
            batch.non_tensor_batch["kl_prcp_weighting"] = np.array(weights)
        else:
            raise NotImplementedError(f"Unknown contrastive KL apply mode: {self.config.algorithm.kl_prcp_apply_mode}.")
        return batch
    
    def _get_kl_prcp_weights_caption(self, batch: DataProto, reward_metrics: Dict[str, Any]) -> DataProto:
        if self.config.algorithm.kl_prcp_apply_mode == "all":
            batch.non_tensor_batch["kl_prcp_weighting_caption"] = np.array([1.0] * len(batch.batch))
        elif self.config.algorithm.kl_prcp_apply_mode == "correct_only":
            weights = []
            for i in range(len(batch.batch)):
                if reward_metrics['accuracy'][i] > 0.1:
                    weights.append(1.0)
                else:
                    weights.append(0.0)
            batch.non_tensor_batch["kl_prcp_weighting_caption"] = np.array(weights)
        elif self.config.algorithm.kl_prcp_apply_mode == "acc_weighted":
            weights = []
            for i in range(len(batch.batch)):
                if reward_metrics['accuracy'][i] > 0.1:
                    weights.append(1.0)
                else:
                    weights.append(self.config.algorithm.incorrect_weighting)
            batch.non_tensor_batch["kl_prcp_weighting_caption"] = np.array(weights)
        else:
            raise NotImplementedError(f"Unknown contrastive KL apply mode: {self.config.algorithm.kl_prcp_apply_mode}.")
        return batch
    
    def _get_correctness_mult_mask(self, batch: DataProto, reward_metrics: Dict[str, Any]) -> DataProto:
        weights = []
        for i in range(len(batch.batch)):
            if reward_metrics['accuracy'][i] > 0.1:
                weights.append(1.0)
            else:
                weights.append(0.0)
        batch.non_tensor_batch["correctness_mult_mask"] = np.array(weights) # len(batch.batch); 0.0 or 1.0
        return batch

    def _get_kl_prcp_coef(self, batch: DataProto) -> DataProto:
        batch.non_tensor_batch["kl_prcp_coef"] = np.array([self.kl_ctrl_contrastive.kl_coef] * len(batch.batch))
        # update
        self.kl_ctrl_contrastive.update(current_kl=None, n_steps=1)
        return batch
    
    def _get_kl_prcp_coef_caption(self, batch: DataProto) -> DataProto:
        batch.non_tensor_batch["kl_prcp_coef_caption"] = np.array([self.kl_ctrl_contrastive_caption.kl_coef] * len(batch.batch))
        # update
        self.kl_ctrl_contrastive_caption.update(current_kl=None, n_steps=1)
        return batch


    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {"min_pixels": self.config.data.min_pixels, "max_pixels": self.config.data.max_pixels}
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)


            if self.config.algorithm.use_caption_rollout:   
                aug_caption_batch_for_rollout = deepcopy(new_batch)
                # aug_caption_batch_for_rollout.batch.pop("responses", None)
                # aug_caption_batch_for_rollout.batch.pop("response_mask", None)
                # aug_caption_batch_for_rollout.batch.pop("prompts", None)
                aug_caption_batch_for_rollout.non_tensor_batch.pop("multi_modal_data", None)
                aug_caption_batch_for_rollout.non_tensor_batch.pop("multi_modal_inputs", None)
                # aug_caption_batch_for_rollout.non_tensor_batch.pop("aug_multi_modal_data", None)
                # aug_caption_batch_for_rollout.non_tensor_batch.pop("aug_multi_modal_inputs", None)
                aug_caption_batch_for_rollout.meta_info.pop("min_pixels", None)
                aug_caption_batch_for_rollout.meta_info.pop("max_pixels", None)
                aug_caption_batch_for_rollout.batch["input_ids"] = aug_caption_batch_for_rollout.batch.pop("aug_caption_input_ids")
                aug_caption_batch_for_rollout.batch["attention_mask"] = aug_caption_batch_for_rollout.batch.pop("aug_caption_attention_mask")
                aug_caption_batch_for_rollout.batch["position_ids"] = aug_caption_batch_for_rollout.batch.pop("aug_caption_position_ids")
                aug_caption_batch_for_rollout.non_tensor_batch["raw_prompt_ids"] = aug_caption_batch_for_rollout.non_tensor_batch.pop("aug_caption_prompt_ids")
                
                # pop those keys for generation
                aug_caption_gen_batch_for_rollout = aug_caption_batch_for_rollout.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                aug_caption_gen_batch_for_rollout_output = self.actor_rollout_ref_wg.generate_sequences(aug_caption_gen_batch_for_rollout)
                aug_caption_batch_for_rollout.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(aug_caption_batch_for_rollout.batch))], dtype=object)
                aug_caption_batch_for_rollout = aug_caption_batch_for_rollout.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                aug_caption_batch_for_rollout = aug_caption_batch_for_rollout.union(aug_caption_gen_batch_for_rollout_output)


            ## get the corrupted images for KL_prcp
            if self.config.algorithm.use_kl_prcp and "multi_modal_data" in new_batch.non_tensor_batch.keys():
                # take the raw PIL images
                aug_multi_modal_data = []
                aug_multi_modal_inputs = []
                for item in new_batch.non_tensor_batch["multi_modal_data"]:
                    if "images_aug" in item:
                        # use pre-augmented images (preprocessed for semantic-aware masking)
                        aug_images_pil = item.pop('images_aug')  # a list
                    else: # online random masking
                        # original_images_pil = item['images'] # a list
                        original_images_pil = item['images_processed'] # a list
                        aug_images_pil = self._aug_img_for_kl_prcp(original_images_pil)

                    aug_multimodal_input = self.processor(aug_images_pil, ["dummy prompt"], add_special_tokens=False, return_tensors="pt")
                    aug_multimodal_input.pop("input_ids")
                    aug_multimodal_input.pop("attention_mask")
                    aug_multi_modal_data.append({"images": aug_images_pil})
                    aug_multi_modal_inputs.append(dict(aug_multimodal_input))
                # add to batch
                new_batch.non_tensor_batch["aug_multi_modal_data"] = aug_multi_modal_data
                new_batch.non_tensor_batch["aug_multi_modal_inputs"] = aug_multi_modal_inputs
                assert new_batch.non_tensor_batch["aug_multi_modal_inputs"][0]['pixel_values'].shape == new_batch.non_tensor_batch["multi_modal_inputs"][0]['pixel_values'].shape

        
            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels"],
            )

            # generate a batch

            # 打印主 batch 的信息
            # print("\n=== 🔷 gen_batch BATCH INFO (gen_batch) ===")
            # print(">> batch.batch:")
            # for k, v in gen_batch.batch.items():
            #     print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")

            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            # 打印主 batch 的信息
            # print("\n=== 🔷 gen_batch_output BATCH INFO (gen_batch_output) ===")
            # print(">> batch.batch:")
            # for k, v in gen_batch_output.batch.items():
            #     print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            new_batch.non_tensor_batch.pop("multi_modal_data", None)
            new_batch.non_tensor_batch.pop("aug_multi_modal_data", None)
            
            if self.config.algorithm.use_caption_rollout:
                new_batch.non_tensor_batch["uid_origin"] = new_batch.non_tensor_batch["uid"]
                aug_caption_batch_for_rollout.non_tensor_batch["uid_origin"] = new_batch.non_tensor_batch["uid"]
            
            if self.config.algorithm.use_kl_prcp_caption:        
                aug_caption_batch = deepcopy(new_batch)
                aug_caption_batch.non_tensor_batch.pop("multi_modal_data", None)
                aug_caption_batch.non_tensor_batch.pop("multi_modal_inputs", None)
                aug_caption_batch.non_tensor_batch.pop("aug_multi_modal_data", None)
                aug_caption_batch.non_tensor_batch.pop("aug_multi_modal_inputs", None)
                aug_caption_batch.batch["input_ids"] = aug_caption_batch.batch.pop("aug_caption_input_ids")
                aug_caption_batch.batch["attention_mask"] = aug_caption_batch.batch.pop("aug_caption_attention_mask")
                aug_caption_batch.batch["position_ids"] = aug_caption_batch.batch.pop("aug_caption_position_ids")
                aug_caption_batch.non_tensor_batch["raw_prompt_ids"] = aug_caption_batch.non_tensor_batch.pop("aug_caption_prompt_ids")

                # ✅ 使用 gen_batch_output 提取 response 段
                gen_responses = gen_batch_output.batch["input_ids"][:, -self.config.data.max_response_length:]
                gen_attention_mask = gen_batch_output.batch["attention_mask"][:, -self.config.data.max_response_length:]
                gen_position_ids = gen_batch_output.batch["position_ids"][:, :,-self.config.data.max_response_length:]

                # ✅ 拼接
                aug_caption_batch.batch["input_ids"] = torch.cat([aug_caption_batch.batch["input_ids"], gen_responses], dim=-1)
                aug_caption_batch.batch["attention_mask"] = torch.cat([aug_caption_batch.batch["attention_mask"], gen_attention_mask], dim=-1)
                aug_caption_batch.batch["position_ids"] = torch.cat([aug_caption_batch.batch["position_ids"], gen_position_ids], dim=-1)

                ######################################################
                if self.config.algorithm.use_caption_rollout:
                    aug_origin_batch = deepcopy(new_batch)
                    # aug_origin_batch.non_tensor_batch.pop("multi_modal_data", None)
                    # aug_origin_batch.non_tensor_batch.pop("multi_modal_inputs", None)
                    aug_origin_batch.non_tensor_batch.pop("aug_multi_modal_data", None)
                    aug_origin_batch.non_tensor_batch.pop("aug_multi_modal_inputs", None)
                    aug_origin_batch.batch.pop("aug_caption_input_ids", None)
                    aug_origin_batch.batch.pop("aug_caption_attention_mask", None)
                    aug_origin_batch.batch.pop("aug_caption_position_ids", None)
                    aug_origin_batch.non_tensor_batch.pop("aug_caption_prompt_ids", None)

                    aug_origin_batch.batch["input_ids"] = gen_batch_output.batch["input_ids"][:, :self.config.data.max_prompt_length]
                    aug_origin_batch.batch["attention_mask"] = gen_batch_output.batch["attention_mask"][:, :self.config.data.max_prompt_length]
                    aug_origin_batch.batch["position_ids"] = gen_batch_output.batch["position_ids"][:,:, :self.config.data.max_prompt_length]
                    # aug_origin_batch.non_tensor_batch["raw_prompt_ids"] = gen_batch_output["raw_prompt_ids"]

                    # ✅ 使用 gen_batch_output 提取 response 段
                    gen_responses = aug_caption_batch_for_rollout.batch["input_ids"][:, -self.config.data.max_response_length:]
                    gen_attention_mask = aug_caption_batch_for_rollout.batch["attention_mask"][:, -self.config.data.max_response_length:]
                    gen_position_ids = aug_caption_batch_for_rollout.batch["position_ids"][:,:, -self.config.data.max_response_length:]

                    # ✅ 拼接
                    aug_origin_batch.batch["input_ids"] = torch.cat([aug_origin_batch.batch["input_ids"], gen_responses], dim=-1)
                    aug_origin_batch.batch["attention_mask"] = torch.cat([aug_origin_batch.batch["attention_mask"], gen_attention_mask], dim=-1)
                    aug_origin_batch.batch["position_ids"] = torch.cat([aug_origin_batch.batch["position_ids"], gen_position_ids], dim=-1)

            new_batch.batch.pop("aug_caption_input_ids", None)
            new_batch.batch.pop("aug_caption_attention_mask", None)
            new_batch.batch.pop("aug_caption_position_ids", None)
            new_batch.non_tensor_batch.pop("aug_caption_prompt_ids", None)    


            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                if self.config.algorithm.use_caption_rollout and self.config.algorithm.use_kl_prcp_caption:
                    return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n], aug_caption_batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n], aug_origin_batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n], aug_caption_batch_for_rollout[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                elif self.config.algorithm.use_caption_rollout:
                    return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n], aug_caption_batch_for_rollout[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                elif self.config.algorithm.use_kl_prcp_caption:
                    return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n], aug_caption_batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                else:
                    return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()

                    if self.config.algorithm.use_caption_rollout and self.config.algorithm.use_kl_prcp_caption:
                        batch, aug_caption_batch, aug_origin_batch, aug_caption_batch_for_rollout = self._make_batch_data(metrics=metrics)
                    elif self.config.algorithm.use_caption_rollout:
                        batch, aug_caption_batch_for_rollout = self._make_batch_data(metrics=metrics)
                    elif self.config.algorithm.use_kl_prcp_caption:
                        batch, aug_caption_batch = self._make_batch_data(metrics=metrics)
                    else:
                        batch = self._make_batch_data(metrics=metrics)

                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                if self.config.algorithm.use_caption_rollout:
                    aug_caption_batch_for_rollout.meta_info["global_token_num"] = torch.sum(aug_caption_batch_for_rollout.batch["attention_mask"], dim=-1).tolist()
                
                # compute reward
                if "token_level_scores" not in batch.batch:
                    reward_ref = self.reward_fn.compute_reward.remote(batch)  # 这个变量后面没用，会被覆盖
                    reward_tensor, reward_metrics = ray.get(reward_ref)
                        

                if self.config.algorithm.use_kl_prcp:
                    # store acc reward in batch
                    batch = self._get_kl_prcp_weights(batch, reward_metrics)
                    # store kl_prcp coef in batch
                    batch = self._get_kl_prcp_coef(batch)
                    
                if self.config.algorithm.use_kl_prcp_caption:
                    # store acc reward in batch
                    batch = self._get_kl_prcp_weights_caption(batch, reward_metrics)
                    # store kl_prcp coef in batch
                    batch = self._get_kl_prcp_coef_caption(batch)

                if self.config.algorithm.use_kl_prcp_caption:
                    # compute log_probs with augmented images
                    with timer("aug_probs_caption", timing_raw):
                        # 在 merge_batch_with_caption_batch 前做

                        # 打印主 batch 的信息
                        print("\n=== 🔷 aug_caption_batch BATCH INFO (use_kl_prcp_caption) ===")
                        print(">> batch.batch:")
                        for k, v in aug_caption_batch.batch.items():
                            print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                        
                        print(">> batch.non_tensor_batch:")
                        for k, v in aug_caption_batch.non_tensor_batch.items():
                            if isinstance(v, list):
                                print(f"  - {k}: list of length {len(v)} (type: {type(v[0]) if v else 'N/A'})")
                            elif isinstance(v, np.ndarray):
                                print(f"  - {k}: np.ndarray of shape {v.shape} dtype {v.dtype}")
                            else:
                                print(f"  - {k}: type {type(v)}")

                        print(">> batch.meta_info:")
                        for k, v in aug_caption_batch.meta_info.items():
                            print(f"  - {k}: {v}")


                        aug_log_probs_caption = self.actor_rollout_ref_wg.compute_log_probs_aug_caption(aug_caption_batch)
                        batch = batch.union(aug_log_probs_caption)


                        # 打印主 batch 的信息
                        print("\n=== 🔷 aug_caption_batch BATCH INFO (use_kl_prcp_caption) ===")
                        print(">> batch.batch:")
                        for k, v in batch.batch.items():
                            print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                        
                        print(">> batch.non_tensor_batch:")
                        for k, v in batch.non_tensor_batch.items():
                            if isinstance(v, list):
                                print(f"  - {k}: list of length {len(v)} (type: {type(v[0]) if v else 'N/A'})")
                            elif isinstance(v, np.ndarray):
                                print(f"  - {k}: np.ndarray of shape {v.shape} dtype {v.dtype}")
                            else:
                                print(f"  - {k}: type {type(v)}")

                        print(">> batch.meta_info:")
                        for k, v in batch.meta_info.items():
                            print(f"  - {k}: {v}")


                        if self.config.algorithm.use_caption_rollout:
                            # Print debug info
                            # print("=== aug_origin_batch Information ===")
                            # print(">> batch.batch:")
                            # for k, v in aug_origin_batch.batch.items():
                            #     print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                            # print(">> non_tensor_batch:")
                            # for k, v in aug_origin_batch.non_tensor_batch.items():
                            #     print(f"  - {k}: {type(v)}")
                            # print(">> meta_info:")
                            # for k, v in aug_origin_batch.meta_info.items():
                            #     print(f"  - {k}: {v}")
                            aug_log_probs_origin = self.actor_rollout_ref_wg.compute_log_probs_aug_caption(aug_origin_batch)
                            aug_caption_batch_for_rollout = aug_caption_batch_for_rollout.union(aug_log_probs_origin)


                # 融合batch和aug_caption_batch，如果key是batch有，而aug_caption_batch没有的，则把aug_caption_batch中的复制一份。
                if self.config.algorithm.use_caption_rollout:
                
                    # 在 merge_batch_with_caption_batch 前做
                    batch.non_tensor_batch["batch_type"] = ["main"] * len(batch)
                    aug_caption_batch_for_rollout.non_tensor_batch["batch_type"] = ["caption"] * len(aug_caption_batch_for_rollout)

                    # # 打印主 batch 的信息
                    # print("\n=== 🔷 MAIN BATCH INFO (before merge) ===")
                    # print(">> batch.batch:")
                    # for k, v in batch.batch.items():
                    #     print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                    
                    # print(">> batch.non_tensor_batch:")
                    # for k, v in batch.non_tensor_batch.items():
                    #     if isinstance(v, list):
                    #         print(f"  - {k}: list of length {len(v)} (type: {type(v[0]) if v else 'N/A'})")
                    #     elif isinstance(v, np.ndarray):
                    #         print(f"  - {k}: np.ndarray of shape {v.shape} dtype {v.dtype}")
                    #     else:
                    #         print(f"  - {k}: type {type(v)}")

                    # print(">> batch.meta_info:")
                    # for k, v in batch.meta_info.items():
                    #     print(f"  - {k}: {v}")

                    # # 打印 caption batch 的信息
                    # print("\n=== 🟣 CAPTION BATCH INFO (before merge) ===")
                    # print(">> aug_caption_batch_for_rollout.batch:")
                    # for k, v in aug_caption_batch_for_rollout.batch.items():
                    #     print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                    
                    # print(">> aug_caption_batch_for_rollout.non_tensor_batch:")
                    # for k, v in aug_caption_batch_for_rollout.non_tensor_batch.items():
                    #     if isinstance(v, list):
                    #         print(f"  - {k}: list of length {len(v)} (type: {type(v[0]) if v else 'N/A'})")
                    #     elif isinstance(v, np.ndarray):
                    #         print(f"  - {k}: np.ndarray of shape {v.shape} dtype {v.dtype}")
                    #     else:
                    #         print(f"  - {k}: type {type(v)}")

                    # print(">> aug_caption_batch_for_rollout.meta_info:")
                    # for k, v in aug_caption_batch_for_rollout.meta_info.items():
                    #     print(f"  - {k}: {v}")

                    # 然后再执行合并
                    batch = merge_batch_with_caption_batch(batch, aug_caption_batch_for_rollout)
                    
                    # # 打印 caption batch 的信息
                    # print("\n=== 🟣 CAPTION BATCH INFO (after merge) ===")
                    # print(">> batch.batch:")
                    # for k, v in batch.batch.items():
                    #     print(f"  - {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                    
                    # print(">> batch.non_tensor_batch:")
                    # for k, v in batch.non_tensor_batch.items():
                    #     if isinstance(v, list):
                    #         print(f"  - {k}: list of length {len(v)} (type: {type(v[0]) if v else 'N/A'})")
                    #     elif isinstance(v, np.ndarray):
                    #         print(f"  - {k}: np.ndarray of shape {v.shape} dtype {v.dtype}")
                    #     else:
                    #         print(f"  - {k}: type {type(v)}")

                    # print(">> batch.meta_info:")
                    # for k, v in batch.meta_info.items():
                    #     print(f"  - {k}: {v}")

                if self.config.algorithm.use_sft_loss or self.config.algorithm.use_sft_loss_2:
                    # compute correctness mask
                    reward_ref = self.reward_fn.compute_reward.remote(batch)  # 这个变量后面没用，会被覆盖
                    reward_tensor, reward_metrics = ray.get(reward_ref)
                    batch = self._get_correctness_mult_mask(batch, reward_metrics)

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor

                        # reduce total reward metrics
                        reduced_reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reduced_reward_metrics)

                        if self.config.algorithm.use_caption_rollout:
                            batch_types = np.array(batch.non_tensor_batch.get("batch_type", ["main"] * len(batch)))

                            reward_metrics_main: Dict[str, List[float]] = {}
                            reward_metrics_caption: Dict[str, List[float]] = {}

                            for k, v in reward_metrics.items():
                                v = np.array(v)
                                reward_metrics_main[k] = v[batch_types == "main"]
                                reward_metrics_caption[k] = v[batch_types == "caption"]

                            reduced_main = reduce_metrics(reward_metrics_main)
                            reduced_caption = reduce_metrics(reward_metrics_caption)

                            metrics.update({f"reward/main_{k}": v for k, v in reduced_main.items()})
                            metrics.update({f"reward/caption_{k}": v for k, v in reduced_caption.items()})

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # self.use_reference_policy_caption = True if "aug_log_probs_caption" in batch.batch else False

                    # # apply kl penalty if available
                    # if self.use_reference_policy_caption:
                    #     # 在修改前先保存一份
                    #     rewards_before = batch.batch.get("token_level_rewards", batch.batch["token_level_scores"]).clone()

                    #     # apply kl penalty to reward
                    #     print("=== Apply KL penalty (caption) ===")
                    #     batch, kl_metrics = apply_kl_penalty_caption(batch, self.kl_ctrl_reward, self.config.algorithm.kl_penalty)
                    #     metrics.update(kl_metrics)

                    #     # 修改后
                    #     rewards_after = batch.batch["token_level_rewards"]

                    #     # 打印前 3 个样本的 response 区域奖励和差异
                    #     mask = batch.batch["response_mask"]
                    #     diff = (rewards_after - rewards_before) * mask
                    #     for i in range(min(3, rewards_before.shape[0])):
                    #         base_sum = (rewards_before[i] * mask[i]).sum().item()
                    #         after_sum = (rewards_after[i] * mask[i]).sum().item()
                    #         delta_sum = (diff[i]).sum().item()
                    #         print(f"[ex#{i}] reward_sum_before={base_sum:.4f}, after={after_sum:.4f}, delta={delta_sum:.4f}")
                    # else:
                    #     batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]



                    # # compute advantages, executed on the driver process
                    # batch = compute_advantage(
                    #     batch,
                    #     adv_estimator=self.config.algorithm.adv_estimator,
                    #     gamma=self.config.algorithm.gamma,
                    #     lam=self.config.algorithm.lam,
                    # )
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        norm_mode="zscore",
                        beta=0.1,
                        w_min=0.5,
                        w_max=1.5,
                        z_max=3.0
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
