# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    T5ForConditionalGeneration,
)

from transformers.trainer import SequentialDistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

from overrides_symbolic import DoNothingDataCollator, T5ForConditionalGenerationCustom
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    duration_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Duration model path"
        },
    )
    eval_model_path: Optional[str] = field(
        default="none",
        metadata={
            "help": "evaluation model"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        inputs_original = []
        inputs_start = []
        inputs_duration_1 = []
        inputs_duration_2 = []

        labels_original = []
        labels_start = []
        labels_duration = []

        end_point_labels = []

        use_logic_losses = []
        use_regular_losses = []

        for l in lines:
            inputs_original.append(l.split("\t")[0])
            inputs_start.append(l.split("\t")[1])
            inputs_duration_1.append(l.split("\t")[2])
            inputs_duration_2.append(l.split("\t")[3])

            labels_original.append(l.split("\t")[4])
            labels_start.append("answer: positive <extra_id_2>")
            labels_duration.append("answer: <extra_id_2>")

            epl = 0
            if "ends after" in l.split("\t")[0]:
                epl = 1
            if "positive" not in l.split("\t")[-1]:
                epl = -100
            end_point_labels.append(epl)

            use_logic_loss = 0
            use_regular_loss = 1
            if "ends after" in l.split("\t")[0] or "ends before" in l.split("\t")[0]:
                use_logic_loss = 1
                use_regular_loss = 0
            use_logic_losses.append(use_logic_loss)
            use_regular_losses.append(use_regular_loss)

        self.inputs_original = tokenizer.batch_encode_plus(inputs_original, pad_to_max_length=True)
        self.inputs_start = tokenizer.batch_encode_plus(inputs_start, pad_to_max_length=True)
        self.inputs_duration_1 = tokenizer.batch_encode_plus(inputs_duration_1, pad_to_max_length=True)
        self.inputs_duration_2 = tokenizer.batch_encode_plus(inputs_duration_2, pad_to_max_length=True)

        self.labels_original = tokenizer.batch_encode_plus(labels_original, pad_to_max_length=True)
        self.labels_start = tokenizer.batch_encode_plus(labels_start, pad_to_max_length=True)
        self.labels_duration = tokenizer.batch_encode_plus(labels_duration, pad_to_max_length=True)

        self.end_point_labels = end_point_labels
        self.use_logic_losses = use_logic_losses
        self.use_regular_losses = use_regular_losses
        assert len(self.use_regular_losses) == len(self.labels_original["input_ids"])
        for i, url in enumerate(self.use_regular_losses):
            if url == 0:
                for j in range(0, 3):
                    if self.labels_original["input_ids"][i][j] != 0:
                        self.labels_original["input_ids"][i][j] = -100
            else:
                self.end_point_labels[i] = -100

    def __len__(self):
        return len(self.inputs_original["input_ids"])

    def __getitem__(self, i):
        return {
            "input_ids_original": self.inputs_original["input_ids"][i],
            "attention_mask_original": self.inputs_original["attention_mask"][i],
            "input_ids_start": self.inputs_start["input_ids"][i],
            "attention_mask_start": self.inputs_start["attention_mask"][i],
            "input_ids_duration_1": self.inputs_duration_1["input_ids"][i],
            "attention_mask_duration_1": self.inputs_duration_1["attention_mask"][i],
            "input_ids_duration_2": self.inputs_duration_2["input_ids"][i],
            "attention_mask_duration_2": self.inputs_duration_2["attention_mask"][i],
            "lm_labels_original": self.labels_original["input_ids"][i],
            "lm_labels_start": self.labels_start["input_ids"][i],
            "lm_labels_duration": self.labels_duration["input_ids"][i],
            "decoder_attention_mask_original": [1] * 3 + [0] * (len(self.labels_original["attention_mask"][i]) - 3),
            "decoder_attention_mask_start": [1] * 4 + [0] * (len(self.labels_start["attention_mask"][i]) - 4),
            "decoder_attention_mask_duration": [1] * 3 + [0] * (len(self.labels_duration["attention_mask"][i]) - 3),
            "use_logic_loss": self.use_logic_losses[i],
            "use_regular_loss": self.use_regular_losses[i],
            "end_point_label": self.end_point_labels[i],
        }


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        ret = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path)
        print("DATA SIZE: ")
        print(len(ret))
        return ret
    else:
        return None


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    duration_model = T5ForConditionalGeneration.from_pretrained(
        model_args.duration_model_path,
    )
    model = T5ForConditionalGenerationCustom.from_pretrained(
        model_args.model_name_or_path,
    )
    model.duration_t5_model = duration_model

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DoNothingDataCollator()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        # trainer.train(model_path=model_path)
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # eval_output = trainer.evaluate()
        if model_args.eval_model_path != "none":
            model = T5ForConditionalGenerationCustom.from_pretrained(model_args.eval_model_path).cuda()
            model.duration_t5_model = duration_model
        else:
            model = T5ForConditionalGenerationCustom.from_pretrained(training_args.output_dir).cuda()
        model.eval()
        sampler = SequentialSampler(eval_dataset)
        data_collator = DoNothingDataCollator()
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=training_args.eval_batch_size,
            collate_fn=data_collator.collate_batch,
        )
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        writer = open(output_eval_file, "w")
        # 2841 -> negative
        # 1465 -> positive
        for inputs in tqdm(data_loader, "Prediction"):
            for k, v in inputs.items():
                inputs[k] = v.cuda()

            with torch.no_grad():
                outputs_lm_logits = model(**inputs)[2].detach().cpu().numpy()
                outputs_end = model(**inputs)[1].detach().cpu().numpy()

                for klm in range(0, len(outputs_lm_logits)):
                    label_1 = "positive"
                    if outputs_lm_logits[klm][2][2841] > outputs_lm_logits[klm][2][1465]:
                        label_1 = "negative"
                    label_2 = str(outputs_end[klm])
                    writer.write("\t".join([label_1, label_2]) + "\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()