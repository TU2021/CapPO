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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        caption_key: Optional[str] = None,
        is_caption_in_format_prompt: bool = True,
        aug_image_key: Optional[str] = None,  # for kl_prcp
        image_dir: Optional[str] = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        format_prompt_caption: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.caption_key = caption_key  # for captioning tasks, we can use this key to store the caption
        self.is_caption_in_format_prompt = is_caption_in_format_prompt
        self.aug_image_key = aug_image_key  # for kl_prcp
        self.image_key = image_key
        self.image_dir = image_dir
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        self.format_prompt_caption = None
        if format_prompt_caption:
            with open(format_prompt_caption, encoding="utf-8") as f:
                self.format_prompt_caption = f.read()

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts, desc="Filtering overlong prompts", num_proc=16
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            if self.caption_key and self.caption_key in example and self.is_caption_in_format_prompt:
                # print(f"Using caption key: {self.caption_key}")
                caption = example[self.caption_key]
                caption_str = caption[0] if isinstance(caption, list) else caption
                # print(type(caption_str), caption_str)
                prompt_str = format_prompt.render(content=prompt_str, caption=caption_str)
                # print(f"Prompt after formatting: {prompt_str}")
            else:
                prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _build_messages_caption(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        # print(self.format_prompt_caption)
        if self.format_prompt_caption:
            format_prompt_caption = Template(self.format_prompt_caption.strip())
            caption = example[self.caption_key]
            caption_str = caption[0] if isinstance(caption, list) else caption
            # print(type(caption_str), caption_str)
            prompt_str = format_prompt_caption.render(content=prompt_str, caption=caption_str)
            # print(f"Prompt after formatting: {prompt_str}")
        return [{"role": "user", "content": prompt_str}]
        
    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key] or []
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            resized_images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
            ] or None
            model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            resized_images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
            ] or None
            model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
            example["multi_modal_data"]["images_processed"] = resized_images
            example["multi_modal_inputs"] = dict(model_inputs)
            # if aug image is provided in the data
            if self.aug_image_key in example:
                aug_images = [process_image(image) for image in example.pop(self.aug_image_key)]
                example["multi_modal_data"]["images_aug"] = aug_images
            # if need to bulid caption
            if self.caption_key and self.caption_key in example and not self.is_caption_in_format_prompt:
                # print("="*40)
                # print("[DEBUG] example keys:", list(example.keys()))
                messages_caption = self._build_messages_caption(example)
                prompt_caption = self.tokenizer.apply_chat_template(messages_caption, add_generation_prompt=True, tokenize=False)
                caption_model_inputs = self.tokenizer([prompt_caption], add_special_tokens=False, return_tensors="pt")
                # example["aug_caption_inputs"] = dict(caption_model_inputs)
                caption_input_ids = caption_model_inputs.pop("input_ids")[0]
                caption_attention_mask = caption_model_inputs.pop("attention_mask")[0]
                if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                    # qwen2vl mrope
                    caption_position_ids = get_rope_index(
                        self.processor,
                        input_ids=caption_input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        attention_mask=caption_attention_mask,
                    )  # (3, seq_length)
                else:
                    caption_position_ids = torch.clip(caption_attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

                caption_input_ids, caption_attention_mask, caption_position_ids = VF.postprocess_data(
                    input_ids=caption_input_ids,
                    attention_mask=caption_attention_mask,
                    position_ids=caption_position_ids,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                caption_raw_prompt_ids = self.tokenizer.encode(prompt_caption, add_special_tokens=False)
                if len(caption_raw_prompt_ids) > self.max_prompt_length:
                    if self.truncation == "left":
                        caption_raw_prompt_ids = caption_raw_prompt_ids[-self.max_prompt_length :]
                    elif self.truncation == "right":
                        caption_raw_prompt_ids = caption_raw_prompt_ids[: self.max_prompt_length]
                    elif self.truncation == "error":
                        raise RuntimeError(f"Prompt length {len(caption_raw_prompt_ids)} is longer than {self.max_prompt_length}.")
                example["aug_caption_input_ids"] = caption_input_ids
                example["aug_caption_attention_mask"] = caption_attention_mask
                example["aug_caption_position_ids"] = caption_position_ids
                example["aug_caption_prompt_ids"] = caption_raw_prompt_ids

        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
