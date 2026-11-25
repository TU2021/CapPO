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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Tuple, Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input

from ...protocol import DataProto
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig
import numpy as np

__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        # multi_modal_inputs = defaultdict(list)
        # if "multi_modal_inputs" in micro_batch:
        #     for input_dict in micro_batch["multi_modal_inputs"]:
        #         for key, value in input_dict.items():
        #             multi_modal_inputs[key].append(value)

        #     for key, value in multi_modal_inputs.items():
        #         if len(value) != 0:
        #             multi_modal_inputs[key] = torch.cat(value, dim=0)
        #         else:
        #             multi_modal_inputs[key] = None

        multi_modal_inputs = defaultdict(list)

        if "multi_modal_inputs" in micro_batch:
            for input_dict in micro_batch["multi_modal_inputs"]:
                if input_dict is None:
                    continue  # 跳过无图数据

                for key, value in input_dict.items():
                    multi_modal_inputs[key].append(value)

            # 将非空值拼接，否则设置为 None
            for key in multi_modal_inputs:
                if len(multi_modal_inputs[key]) > 0:
                    multi_modal_inputs[key] = torch.cat(multi_modal_inputs[key], dim=0)
                else:
                    multi_modal_inputs[key] = None

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys.extend(["old_log_probs", "advantages"])
        non_tensor_select_keys = ["batch_type"]

        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("multi_modal_inputs")

        # for contrastive kl
        if "aug_log_probs" in data.batch.keys() and self.config.use_kl_prcp:
            select_keys.append("aug_log_probs")
            non_tensor_select_keys.append("kl_prcp_weighting")
            non_tensor_select_keys.append("kl_prcp_coef")
        
        # for contrastive kl
        if "aug_log_probs_caption" in data.batch.keys() and self.config.use_kl_prcp_caption:
            select_keys.append("aug_log_probs_caption")
            non_tensor_select_keys.append("kl_prcp_weighting_caption")
            non_tensor_select_keys.append("kl_prcp_coef_caption")

        if self.config.use_sft_loss or self.config.use_sft_loss_2 :
            non_tensor_select_keys.append("correctness_mult_mask")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            mini_batches_iter = tqdm(mini_batches, desc="Train mini-batches", position=1) if self.rank == 0 else mini_batches
            for mini_batch in mini_batches_iter:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                micro_batches_iter = tqdm(micro_batches, desc="Update policy", position=2) if self.rank == 0 else micro_batches

                for micro_batch in micro_batches_iter:
                    # 分离 main 与 caption
                    batch_type = np.array(micro_batch.non_tensor_batch.get("batch_type", ["main"] * len(micro_batch)))
                    main_idxs = np.where(batch_type == "main")[0]
                    caption_idxs = np.where(batch_type == "caption")[0]
                    # print("main:", main_idxs)
                    # print("caption:", caption_idxs)

                    total_loss = 0.0
                    total_metrics = {}

                    # 主 batch（始终存在）
                    if main_idxs.size > 0:
                        main_batch = micro_batch[main_idxs]
                        pg_loss_main, metrics_main = self.handle_main_batch(main_batch, temperature)
                        loss_main = pg_loss_main / (self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update)
                        total_loss += loss_main
                        append_to_dict(total_metrics, metrics_main)

                    # caption batch（可选存在）
                    if caption_idxs.size > 0:
                        caption_batch = micro_batch[caption_idxs]
                        pg_loss_caption, metrics_caption = self.handle_caption_batch(caption_batch, temperature)
                        loss_caption = pg_loss_caption / (self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update)
                        total_loss += loss_caption
                        append_to_dict(total_metrics, metrics_caption)

                    # backward 两个 loss 之和
                    total_loss.backward()

                    # log
                    append_to_dict(metrics, total_metrics)

                grad_norm = self._optimizer_step()
                metrics["actor/grad_norm"] = grad_norm.item()

        return metrics


    def handle_main_batch(self, batch: DataProto, temperature: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        model_inputs = {**batch.batch, **batch.non_tensor_batch}
        responses = model_inputs["responses"]
        # print("responses:", responses)
        response_mask = model_inputs["attention_mask"][:, -responses.size(1):]
        old_log_probs = model_inputs["old_log_probs"]
        advantages = model_inputs["advantages"]
        metrics = {}

        log_probs = self._forward_micro_batch(model_inputs, temperature)
        entropy_loss = -VF.masked_mean(log_probs, response_mask)

        pg_loss, pg_metrics = compute_policy_loss(
            old_log_probs, log_probs, advantages, response_mask,
            self.config.clip_ratio_low,
            self.config.clip_ratio_high,
            self.config.clip_ratio_dual,
            self.config.loss_avg_mode,
        )

        metrics.update({
            "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
            "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
            "actor/entropy_loss": pg_metrics["entropy_loss"],
            "actor/ppo_kl": pg_metrics["ppo_kl"],
        })

        if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
            ref_log_probs = model_inputs["ref_log_probs"]
            kld = compute_kl(log_probs, ref_log_probs, self.config.kl_penalty)
            kl_loss = average_loss(kld, response_mask, self.config.loss_avg_mode)
            pg_loss += self.config.kl_coef * kl_loss
            metrics["actor/kl_loss"] = kl_loss.item()
            metrics["actor/kl_coef"] = self.config.kl_coef

        if self.config.use_kl_prcp and "aug_log_probs" in model_inputs:
            aug_log_probs = model_inputs["aug_log_probs"]
            aug_kld = compute_kl(log_probs, aug_log_probs, self.config.kl_prcp_penalty)
            weighting = torch.tensor(batch.non_tensor_batch["kl_prcp_weighting"], device=aug_kld.device).unsqueeze(1)
            aug_kld *= weighting
            if self.config.use_kl_prcp_token_level_mask:
                mask = response_mask.bool()
                masked_kld = aug_kld.masked_fill(~mask, float("nan"))
                thresh = torch.nanquantile(masked_kld, 1.0 - self.config.kl_prcp_token_level_mask_top_p, dim=1, keepdim=True)
                token_mask = aug_kld >= thresh
                aug_kld *= token_mask.to(aug_kld.dtype)
            if self.config.use_kl_prcp_clipping:
                aug_kld = torch.clamp(aug_kld, 0.0, self.config.kl_prcp_clipping)
            kl_prcp_loss = VF.masked_mean(aug_kld, response_mask)
            coef = batch.non_tensor_batch.get("kl_prcp_coef", [self.config.kl_prcp_coef])[0]
            pg_loss -= coef * kl_prcp_loss
            metrics["actor/kl_prcp_loss"] = -kl_prcp_loss.item()
            metrics["actor/kl_prcp_coef"] = coef
            if self.config.use_aug_entropy_loss:
                aug_entropy_loss = -VF.masked_mean(aug_log_probs, response_mask)
                pg_loss += self.config.aug_entropy_loss_coef * aug_entropy_loss
                metrics["actor/aug_entropy_loss"] = aug_entropy_loss.item()

        if self.config.use_kl_prcp_caption and "aug_log_probs_caption" in model_inputs and self.config.use_kl_prcp_caption_1:
            aug_log_probs_caption = model_inputs["aug_log_probs_caption"]
            aug_kld_caption = compute_kl(log_probs, aug_log_probs_caption, self.config.kl_prcp_penalty_caption)
            weighting = torch.tensor(batch.non_tensor_batch["kl_prcp_weighting_caption"], device=aug_kld_caption.device).unsqueeze(1)
            aug_kld_caption *= weighting
            if self.config.use_kl_prcp_token_level_mask:
                mask = response_mask.bool()
                masked_kld = aug_kld_caption.masked_fill(~mask, float("nan"))
                thresh = torch.nanquantile(masked_kld, 1.0 - self.config.kl_prcp_token_level_mask_top_p, dim=1, keepdim=True)
                token_mask = aug_kld_caption >= thresh
                aug_kld_caption *= token_mask.to(aug_kld_caption.dtype)
            if self.config.use_kl_prcp_clipping:
                aug_kld_caption = torch.clamp(aug_kld_caption, 0.0, self.config.kl_prcp_clipping)
            kl_caption = VF.masked_mean(aug_kld_caption, response_mask)
            coef = batch.non_tensor_batch.get("kl_prcp_coef_caption", [self.config.kl_prcp_coef_caption])[0]
            pg_loss += coef * kl_caption
            metrics["actor/kl_prcp_loss_caption"] = kl_caption.item()
            metrics["actor/kl_prcp_coef_caption"] = coef
            if self.config.use_aug_entropy_loss_caption:
                aug_entropy_loss_caption = -VF.masked_mean(aug_log_probs_caption, response_mask)
                pg_loss += self.config.aug_entropy_loss_coef_caption * aug_entropy_loss_caption
                metrics["actor/aug_entropy_loss_caption"] = aug_entropy_loss_caption.item()

        if self.config.use_ori_entropy_loss:
            pg_loss += self.config.ori_entropy_loss_coef * entropy_loss
            metrics["actor/ori_entropy_loss"] = entropy_loss.item()
            metrics["actor/ori_entropy_loss_coef"] = self.config.ori_entropy_loss_coef

        if self.config.use_sft_loss:
            correctness_mask = torch.tensor(batch.non_tensor_batch["correctness_mult_mask"], device=log_probs.device).unsqueeze(1)
            sft_loss = VF.masked_mean(-log_probs, response_mask * correctness_mask)
            pg_loss += self.config.sft_loss_coef * sft_loss
            metrics["actor/sft_loss"] = sft_loss.item()
            metrics["actor/sft_coef"] = self.config.sft_loss_coef

        metrics["actor/pg_loss"] = pg_loss.item()
        return pg_loss, metrics


    def handle_caption_batch(self, batch: DataProto, temperature: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        model_inputs = {**batch.batch, **batch.non_tensor_batch}
        responses = model_inputs["responses"]
        response_mask = model_inputs["attention_mask"][:, -responses.size(1):]
        old_log_probs = model_inputs["old_log_probs"]
        advantages = model_inputs["advantages"]
        metrics = {}

        log_probs = self._forward_micro_batch(model_inputs, temperature)
        entropy_loss = -VF.masked_mean(log_probs, response_mask)

        pg_loss, pg_metrics = compute_policy_loss(
            old_log_probs, log_probs, advantages, response_mask,
            self.config.clip_ratio_low,
            self.config.clip_ratio_high,
            self.config.clip_ratio_dual,
            self.config.loss_avg_mode,
        )

        metrics.update({
            "actor_caption/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
            "actor_caption/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
            "actor_caption/entropy_loss": pg_metrics["entropy_loss"],
            "actor_caption/ppo_kl": pg_metrics["ppo_kl"],
        })

        if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
            ref_log_probs = model_inputs["ref_log_probs"]
            kld = compute_kl(log_probs, ref_log_probs, self.config.kl_penalty)
            kl_loss = average_loss(kld, response_mask, self.config.loss_avg_mode)
            pg_loss += self.config.kl_coef * kl_loss
            metrics["actor_caption/kl_loss"] = kl_loss.item()
            metrics["actor_caption/kl_coef"] = self.config.kl_coef

        if self.config.use_kl_prcp_caption and "aug_log_probs_caption" in model_inputs and self.config.use_kl_prcp_caption_2:
            aug_log_probs_caption = model_inputs["aug_log_probs_caption"]
            aug_kld_caption = compute_kl(log_probs, aug_log_probs_caption, self.config.kl_prcp_penalty_caption)
            weighting = torch.tensor(batch.non_tensor_batch["kl_prcp_weighting_caption"], device=aug_kld_caption.device).unsqueeze(1)
            aug_kld_caption *= weighting
            if self.config.use_kl_prcp_token_level_mask:
                mask = response_mask.bool()
                masked_kld = aug_kld_caption.masked_fill(~mask, float("nan"))
                thresh = torch.nanquantile(masked_kld, 1.0 - self.config.kl_prcp_token_level_mask_top_p, dim=1, keepdim=True)
                token_mask = aug_kld_caption >= thresh
                aug_kld_caption *= token_mask.to(aug_kld_caption.dtype)
            if self.config.use_kl_prcp_clipping:
                aug_kld_caption = torch.clamp(aug_kld_caption, 0.0, self.config.kl_prcp_clipping)
            kl_caption = VF.masked_mean(aug_kld_caption, response_mask)
            coef = batch.non_tensor_batch.get("kl_prcp_coef_caption", [self.config.kl_prcp_coef_caption])[0]
            pg_loss += coef * kl_caption
            metrics["actor_caption/kl_prcp_loss_caption"] = kl_caption.item()
            metrics["actor_caption/kl_prcp_coef_caption"] = coef
            if self.config.use_aug_entropy_loss_caption:
                aug_entropy_loss_caption = -VF.masked_mean(aug_log_probs_caption, response_mask)
                pg_loss += self.config.aug_entropy_loss_coef_caption * aug_entropy_loss_caption
                metrics["actor_caption/aug_entropy_loss_caption"] = aug_entropy_loss_caption.item()

        if self.config.use_ori_entropy_loss:
            pg_loss += self.config.ori_entropy_loss_coef * entropy_loss
            metrics["actor_caption/ori_entropy_loss"] = entropy_loss.item()
            metrics["actor_caption/ori_entropy_loss_coef"] = self.config.ori_entropy_loss_coef

        if self.config.use_sft_loss:
            correctness_mask = torch.tensor(batch.non_tensor_batch["correctness_mult_mask"], device=log_probs.device).unsqueeze(1)
            sft_loss = VF.masked_mean(-log_probs, response_mask * correctness_mask)
            pg_loss += self.config.sft_loss_coef * sft_loss
            metrics["actor_caption/sft_loss"] = sft_loss.item()
            metrics["actor_caption/sft_coef"] = self.config.sft_loss_coef

        if self.config.use_sft_loss_2:
            aug_log_probs_caption = model_inputs["aug_log_probs_caption"]
            correctness_mask = torch.tensor(batch.non_tensor_batch["correctness_mult_mask"], device=log_probs.device).unsqueeze(1)
            if self.config.use_entropy_coef:
                entropy_aug_log_probs_caption =  aug_log_probs_caption * torch.exp(log_probs).detach()
                sft_loss = VF.masked_mean(-entropy_aug_log_probs_caption, response_mask * correctness_mask)
            else:
                sft_loss = VF.masked_mean(-aug_log_probs_caption, response_mask * correctness_mask)
            pg_loss += self.config.sft_loss_coef * sft_loss
            metrics["actor_caption/sft_loss_2"] = sft_loss.item()
            metrics["actor_caption/sft_coef_2"] = self.config.sft_loss_coef

        metrics["actor_caption/pg_loss"] = pg_loss.item()
        return pg_loss, metrics
