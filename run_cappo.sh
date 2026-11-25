#!/usr/bin/env bash
# Strict shared-code / original-runtime CapPO baseline, using the full cleaned data.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(command -v "${CAPPO_PYTHON:-python3}")" || { echo 'Set CAPPO_PYTHON to the training environment Python.' >&2; exit 2; }
MODEL=${CAPPO_MODEL_PATH:?Set CAPPO_MODEL_PATH to the base model directory}
TRAIN_FILE=${CAPPO_TRAIN_FILE:?Set CAPPO_TRAIN_FILE to the training JSONL}
TRAIN_IMAGE_DIR=${CAPPO_TRAIN_IMAGE_DIR:?Set CAPPO_TRAIN_IMAGE_DIR to the training image root}
VAL_FILE=${CAPPO_VAL_FILE:?Set CAPPO_VAL_FILE to the validation JSONL}
VAL_IMAGE_DIR=${CAPPO_VAL_IMAGE_DIR:?Set CAPPO_VAL_IMAGE_DIR to the validation image root}
RUN_NAME=CapPO_Qwen2.5_VL_7B_step150
GPUS=${CAPPO_GPUS:-0,1,2,3,4,5,6,7}
EPOCHS=3
MAX_STEPS=150
SAVE_DIR=${CAPPO_CHECKPOINT_DIR:-$REPO/runs/$RUN_NAME/checkpoints}
RAY_DIR=${CAPPO_RAY_TMPDIR:-/tmp/cappo-ray-$UID}

for file in "$TRAIN_FILE" "$VAL_FILE" "$REPO/examples/config.yaml"; do
  [[ -f "$file" ]] || { printf 'Missing file: %s\n' "$file" >&2; exit 2; }
done
for dir in "$MODEL" "$TRAIN_IMAGE_DIR/images" "$VAL_IMAGE_DIR"; do
  [[ -d "$dir" ]] || { printf 'Missing directory: %s\n' "$dir" >&2; exit 2; }
done

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="$GPUS" RAY_TMPDIR="$RAY_DIR"

CMD=("$PYTHON" -m verl.trainer.main config=examples/config.yaml
  "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE" "data.image_dir=$TRAIN_IMAGE_DIR" "data.val_image_dir=$VAL_IMAGE_DIR"
  data.format_prompt=examples/format_prompt/math_format_perception.jinja data.format_prompt_caption=examples/format_prompt/caption_answer.jinja
  data.is_caption_in_format_prompt=false data.caption_key=caption data.prompt_key=problem data.seed=1 data.shuffle=true data.filter_overlong_prompts=true
  data.min_pixels=200704 data.max_pixels=1003520 data.max_prompt_length=4096 data.max_response_length=8192 data.rollout_batch_size=512
  worker.actor.global_batch_size=128 worker.actor.micro_batch_size_per_device_for_update=4 worker.actor.micro_batch_size_per_device_for_experience=16
  worker.actor.model.freeze_vision_tower=false "worker.actor.model.model_path=$MODEL"
  worker.rollout.n=8 worker.rollout.tensor_parallel_size=2 worker.rollout.gpu_memory_utilization=0.6 worker.rollout.max_num_batched_tokens=20000 worker.rollout.val_override_config.temperature=1.0
  worker.reward.reward_function=examples/reward_function/math_robust.py:compute_score
  algorithm.online_filtering=false algorithm.disable_kl=true
  algorithm.use_ori_entropy_loss=true algorithm.ori_entropy_loss_coef=0.05 algorithm.use_aug_entropy_loss_caption=true algorithm.aug_entropy_loss_coef_caption=0.05
  algorithm.use_kl_prcp=false algorithm.use_caption_rollout=false algorithm.use_kl_prcp_caption=true algorithm.use_kl_prcp_caption_1=true algorithm.use_kl_prcp_caption_2=false
  algorithm.kl_prcp_penalty_caption=kl algorithm.kl_prcp_coef=0.01 algorithm.kl_prcp_coef_caption=0.01
  algorithm.use_sft_loss_2=false algorithm.sft_loss_coef=0.5 algorithm.use_entropy_coef=false
  trainer.project_name=CapPO "trainer.experiment_name=$RUN_NAME" 'trainer.logger=["console"]' trainer.nnodes=1 trainer.n_gpus_per_node=8
  "trainer.total_epochs=$EPOCHS" trainer.val_freq=10 trainer.save_freq=10 trainer.save_limit=2 trainer.save_model_only=false trainer.val_before_train=false "trainer.save_checkpoint_path=$SAVE_DIR")
[[ -z $MAX_STEPS ]] || CMD+=("trainer.max_steps=$MAX_STEPS")

echo "[CapPO] shared-source fixed baseline | vLLM 0.9.1 | full 36,579 records | epochs=$EPOCHS"
echo "[CapPO] alpha=0.01 fixed; beta=0.10 is the shared-code advantage setting; checkpoints=$SAVE_DIR"
[[ ${CAPPO_PREFLIGHT_ONLY:-false} != true ]] || { echo '[CapPO] preflight passed'; exit 0; }
mkdir -p -- "$SAVE_DIR" "$RAY_DIR"
cd "$REPO"
exec "${CMD[@]}"
