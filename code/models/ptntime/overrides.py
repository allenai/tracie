from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.data_collator import DataCollator


@dataclass
class DoNothingDataCollator(DataCollator):
    def collate_batch(self, features) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        lm_labels = torch.tensor([f['lm_labels'] for f in features], dtype=torch.long)
        decoder_attention_mask = torch.tensor([f['decoder_attention_mask'] for f in features], dtype=torch.long)
        decoder_inputs = torch.tensor([0, 1525, 10], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }

@dataclass
class DoNothingDataCollatorForGeneration(DataCollator):
    def collate_batch(self, features) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        lm_labels = torch.tensor([f['lm_labels'] for f in features], dtype=torch.long)
        decoder_attention_mask = []
        decoder_inputs = []
        for i in range(0, len(features)):
            decoder_inputs.append([0, 1525, 10] + [0] * (input_ids.size()[1] - 3))
            decoder_attention_mask.append([1, 1, 1, 1] + [0] * (input_ids.size()[1] - 4))
        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
        decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs,
            "lm_labels": lm_labels,
        }

@dataclass
class DataCollatorForLanguageModeling(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "masked_lm_labels": labels}
        else:
            return {"input_ids": batch, "labels": batch}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, 0.0)
        storage = []
        for slice in labels:
            cur = 0
            for val in slice:
                if val == self.tokenizer.pad_token_id:
                    break
                cur += 1
            storage.append(cur)
        for i in range(0, probability_matrix.shape[0]):
            probability_matrix[i, storage[i] - 8 : storage[i]] = 0.85
        # special_tokens_mask = [
        #     self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        # ]
        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # print(masked_indices[0])
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
