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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple, Dict, Any

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    aug_image_key: Optional[str] = None
    response_key: str = "responses"
    caption_key: Optional[str] = None
    is_caption_in_format_prompt: bool = True
    image_dir: Optional[str] = None
    val_image_dir: Optional[str] = None
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    format_prompt_caption: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    # For Evaluation
    n_samples: int = 1
    output_path: Optional[str] = None

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None

        if self.format_prompt_caption is not None:
            if os.path.exists(self.format_prompt_caption):  # ray job uses absolute path
                self.format_prompt_caption = os.path.abspath(self.format_prompt_caption)
            else:
                print(f"Format prompt file {self.format_prompt_caption} not found.")
                self.format_prompt_caption = None

@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""

    # The following are for PAPO
    use_kl_prcp: bool = False
    contrastive_type: str = "augmented"
    kl_prcp_schedule: str = "fixed"
    kl_prcp_schedule_args: Dict[str, Any] = field(default_factory=dict)
    kl_prcp_penalty: str = "kl"
    kl_prcp_coef: float = 1e-3
    kl_prcp_apply_mode: str = "all"
    aug_config: Dict[str, Any] = field(default_factory=dict)

    incorrect_weighting: float = 0.1
    # Double Entropy Loss
    use_aug_entropy_loss: bool = False
    aug_entropy_loss_coef: float = 0.03

    use_ori_entropy_loss: bool = False
    ori_entropy_loss_coef: float = 0.03
    
    # kl_prcp_caption
    use_kl_prcp_caption: bool = False
    contrastive_type_caption: str = "augmented"
    kl_prcp_schedule_caption: str = "fixed"
    kl_prcp_schedule_args_caption: Dict[str, Any] = field(default_factory=dict)
    kl_prcp_penalty_caption: str = "kl"
    kl_prcp_coef_caption: float = 1e-3
    kl_prcp_apply_mode_caption: str = "all"
    aug_config_caption: Dict[str, Any] = field(default_factory=dict)

    use_kl_prcp_caption_1: bool = False
    use_kl_prcp_caption_2: bool = False

    # Double Entropy Loss
    use_aug_entropy_loss_caption: bool = False
    aug_entropy_loss_coef_caption: float = 0.03

    # caption_rollout
    use_caption_rollout: bool = False

    # Other experimental settings    
    use_kl_prcp_clipping: bool = False
    kl_prcp_clipping: float = 0.2

    use_kl_prcp_token_level_mask: bool = False
    kl_prcp_token_level_mask_top_p: float = 0.2

    use_sft_loss: bool = False
    sft_loss_coef: float = 1e-3
    use_sft_loss_2: bool = False
    use_entropy_coef: bool = False



@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = False
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.rollout.response_key = self.data.response_key
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.use_kl_prcp = self.algorithm.use_kl_prcp
        self.worker.actor.kl_prcp_penalty = self.algorithm.kl_prcp_penalty
        self.worker.actor.kl_prcp_coef = self.algorithm.kl_prcp_coef
        self.worker.actor.kl_prcp_apply_mode = self.algorithm.kl_prcp_apply_mode
        self.worker.actor.use_aug_entropy_loss = self.algorithm.use_aug_entropy_loss
        self.worker.actor.aug_entropy_loss_coef = self.algorithm.aug_entropy_loss_coef
        self.worker.actor.use_ori_entropy_loss = self.algorithm.use_ori_entropy_loss
        self.worker.actor.ori_entropy_loss_coef = self.algorithm.ori_entropy_loss_coef
        self.worker.actor.use_kl_prcp_clipping = self.algorithm.use_kl_prcp_clipping
        self.worker.actor.kl_prcp_clipping = self.algorithm.kl_prcp_clipping
        self.worker.actor.use_kl_prcp_token_level_mask = self.algorithm.use_kl_prcp_token_level_mask
        self.worker.actor.kl_prcp_token_level_mask_top_p = self.algorithm.kl_prcp_token_level_mask_top_p
        # keys for kl_caption and caption_rollout
        self.worker.actor.use_kl_prcp_caption = self.algorithm.use_kl_prcp_caption
        self.worker.actor.use_kl_prcp_caption_1 = self.algorithm.use_kl_prcp_caption_1
        self.worker.actor.use_kl_prcp_caption_2 = self.algorithm.use_kl_prcp_caption_2
        self.worker.actor.use_caption_rollout = self.algorithm.use_caption_rollout
        self.worker.actor.kl_prcp_penalty_caption = self.algorithm.kl_prcp_penalty_caption
        self.worker.actor.kl_prcp_coef_caption = self.algorithm.kl_prcp_coef_caption
        self.worker.actor.kl_prcp_apply_mode_caption = self.algorithm.kl_prcp_apply_mode_caption
        self.worker.actor.use_aug_entropy_loss_caption = self.algorithm.use_aug_entropy_loss_caption
        self.worker.actor.aug_entropy_loss_coef_caption = self.algorithm.aug_entropy_loss_coef_caption
        # auto keys for sft loss
        self.worker.actor.use_sft_loss = self.algorithm.use_sft_loss
        self.worker.actor.sft_loss_coef = self.algorithm.sft_loss_coef
        self.worker.actor.use_sft_loss_2 = self.algorithm.use_sft_loss_2
        use_entropy_coef = self.algorithm.use_entropy_coef
        
    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
