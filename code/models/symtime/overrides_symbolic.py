from dataclasses import dataclass
import copy
import logging
from typing import Any, Dict, List, NewType, Tuple

import torch
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import DataCollator
from transformers.modeling_t5 import T5PreTrainedModel, T5Stack, T5ForConditionalGeneration

@dataclass
class DoNothingDataCollator(DataCollator):
    def collate_batch(self, features) -> Dict[str, torch.Tensor]:
        input_ids_original = torch.tensor([f['input_ids_original'] for f in features], dtype=torch.long)
        input_ids_start = torch.tensor([f['input_ids_start'] for f in features], dtype=torch.long)
        input_ids_duration_1 = torch.tensor([f['input_ids_duration_1'] for f in features], dtype=torch.long)
        input_ids_duration_2 = torch.tensor([f['input_ids_duration_2'] for f in features], dtype=torch.long)
        attention_mask_original = torch.tensor([f['attention_mask_original'] for f in features], dtype=torch.long)
        attention_mask_start = torch.tensor([f['attention_mask_start'] for f in features], dtype=torch.long)
        attention_mask_duration_1 = torch.tensor([f['attention_mask_duration_1'] for f in features], dtype=torch.long)
        attention_mask_duration_2 = torch.tensor([f['attention_mask_duration_2'] for f in features], dtype=torch.long)
        lm_labels_original = torch.tensor([f['lm_labels_original'] for f in features], dtype=torch.long)
        lm_labels_start = torch.tensor([f['lm_labels_start'] for f in features], dtype=torch.long)
        lm_labels_duration = torch.tensor([f['lm_labels_duration'] for f in features], dtype=torch.long)
        decoder_attention_mask_original = torch.tensor([f['decoder_attention_mask_original'] for f in features], dtype=torch.long)
        decoder_attention_mask_start = torch.tensor([f['decoder_attention_mask_start'] for f in features], dtype=torch.long)
        decoder_attention_mask_duration = torch.tensor([f['decoder_attention_mask_duration'] for f in features], dtype=torch.long)
        use_logic_loss = torch.tensor([f['use_logic_loss'] for f in features], dtype=torch.long)
        use_regular_loss = torch.tensor([f['use_regular_loss'] for f in features], dtype=torch.long)
        end_point_label = torch.tensor([f['end_point_label'] for f in features], dtype=torch.long)

        return {
            "input_ids_original": input_ids_original,
            "input_ids_start": input_ids_start,
            "input_ids_duration_1": input_ids_duration_1,
            "input_ids_duration_2": input_ids_duration_2,
            "attention_mask_original": attention_mask_original,
            "attention_mask_start": attention_mask_start,
            "attention_mask_duration_1": attention_mask_duration_1,
            "attention_mask_duration_2": attention_mask_duration_2,
            "lm_labels_original": lm_labels_original,
            "lm_labels_start": lm_labels_start,
            "lm_labels_duration": lm_labels_duration,
            "decoder_attention_mask_original": decoder_attention_mask_original,
            "decoder_attention_mask_start": decoder_attention_mask_start,
            "decoder_attention_mask_duration": decoder_attention_mask_duration,
            "use_logic_loss": use_logic_loss,
            "use_regular_loss": use_regular_loss,
            "end_point_label": end_point_label,
        }


class T5ForConditionalGenerationCustom(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.duration_t5_model = T5ForConditionalGeneration(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.s = nn.Softmax(dim=1)
        self.r = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        self.discrete_value_ids = [32000, 32001, 32002, 32003, 32004, 32005, 32006]

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids_original=None,
        input_ids_start=None,
        input_ids_duration_1=None,
        input_ids_duration_2=None,
        attention_mask_original=None,
        attention_mask_start=None,
        attention_mask_duration_1=None,
        attention_mask_duration_2=None,
        decoder_attention_mask_original=None,
        decoder_attention_mask_start=None,
        decoder_attention_mask_duration=None,
        lm_labels_original=None,
        lm_labels_start=None,
        lm_labels_duration=None,
        end_point_label=None,
        use_logic_loss=None,
        use_regular_loss=None,
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.

    Examples::

        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model.generate(input_ids)
        """

        encoder_outputs_original = self.encoder(
            input_ids=input_ids_original, attention_mask=attention_mask_original
        )
        encoder_outputs_start = self.encoder(
            input_ids=input_ids_start, attention_mask=attention_mask_start
        )
        encoder_outputs_duration_1 = self.duration_t5_model.encoder(
            input_ids=input_ids_duration_1, attention_mask=attention_mask_duration_1
        )
        encoder_outputs_duration_2 = self.duration_t5_model.encoder(
            input_ids=input_ids_duration_2, attention_mask=attention_mask_duration_2
        )

        hidden_states_original = encoder_outputs_original[0]
        hidden_states_start = encoder_outputs_start[0]
        hidden_states_duration_1 = encoder_outputs_duration_1[0]
        hidden_states_duration_2 = encoder_outputs_duration_2[0]

        decoder_input_ids_original = self._shift_right(lm_labels_original)
        decoder_input_ids_start = self._shift_right(lm_labels_start)
        decoder_input_ids_duration = self._shift_right(lm_labels_duration)

        decoder_outputs_original = self.decoder(
            input_ids=decoder_input_ids_original,
            attention_mask=decoder_attention_mask_original,
            encoder_hidden_states=hidden_states_original,
            encoder_attention_mask=attention_mask_original,
            use_cache=False,
        )
        decoder_outputs_start = self.decoder(
            input_ids=decoder_input_ids_start,
            attention_mask=decoder_attention_mask_start,
            encoder_hidden_states=hidden_states_start,
            encoder_attention_mask=attention_mask_start,
            use_cache=False,
        )

        decoder_outputs_duration_1 = self.duration_t5_model.decoder(
            input_ids=decoder_input_ids_duration,
            attention_mask=decoder_attention_mask_duration,
            encoder_hidden_states=hidden_states_duration_1,
            encoder_attention_mask=attention_mask_duration_1,
            use_cache=False,
        )

        decoder_outputs_duration_2 = self.duration_t5_model.decoder(
            input_ids=decoder_input_ids_duration,
            attention_mask=decoder_attention_mask_duration,
            encoder_hidden_states=hidden_states_duration_2,
            encoder_attention_mask=attention_mask_duration_2,
            use_cache=False,
        )

        sequence_output_original = decoder_outputs_original[0]
        sequence_output_original = sequence_output_original * (self.model_dim ** -0.5)
        lm_logits_original = self.lm_head(sequence_output_original)

        sequence_output_start = decoder_outputs_start[0]
        sequence_output_start = sequence_output_start * (self.model_dim ** -0.5)
        lm_logits_start = self.lm_head(sequence_output_start)

        sequence_output_duration_1 = decoder_outputs_duration_1[0]
        sequence_output_duration_1 = sequence_output_duration_1 * (self.model_dim ** -0.5)
        lm_logits_duration_1 = self.duration_t5_model.lm_head(sequence_output_duration_1)

        sequence_output_duration_2 = decoder_outputs_duration_2[0]
        sequence_output_duration_2 = sequence_output_duration_2 * (self.model_dim ** -0.5)
        lm_logits_duration_2 = self.duration_t5_model.lm_head(sequence_output_duration_2)

        start_before_after_factor = lm_logits_start[:, 2, 1465].view(-1, 1) - lm_logits_start[:, 2, 2841].view(-1, 1)
        # Make sure the factor is either -1 or 1.
        start_before_after_factor = torch.tanh(start_before_after_factor * 10000.0)

        dist = torch.cat((
            lm_logits_start[:, 3, self.discrete_value_ids[0]].view(-1, 1),
            lm_logits_start[:, 3, self.discrete_value_ids[1]].view(-1, 1),
            lm_logits_start[:, 3, self.discrete_value_ids[2]].view(-1, 1),
            lm_logits_start[:, 3, self.discrete_value_ids[3]].view(-1, 1),
            lm_logits_start[:, 3, self.discrete_value_ids[4]].view(-1, 1),
            lm_logits_start[:, 3, self.discrete_value_ids[5]].view(-1, 1),
            lm_logits_start[:, 3, self.discrete_value_ids[6]].view(-1, 1),
        ), 1)
        dist = self.s(dist)

        constant_vec = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6]).cuda().repeat(dist.size(0), 1).view(-1, 7, 1)

        dist = torch.bmm(dist.view(-1, 1, 7), constant_vec).view(-1, 1) * start_before_after_factor

        duration_val_1 = torch.cat((
            lm_logits_duration_1[:, 2, self.discrete_value_ids[0]].view(-1, 1),
            lm_logits_duration_1[:, 2, self.discrete_value_ids[1]].view(-1, 1),
            lm_logits_duration_1[:, 2, self.discrete_value_ids[2]].view(-1, 1),
            lm_logits_duration_1[:, 2, self.discrete_value_ids[3]].view(-1, 1),
            lm_logits_duration_1[:, 2, self.discrete_value_ids[4]].view(-1, 1),
            lm_logits_duration_1[:, 2, self.discrete_value_ids[5]].view(-1, 1),
            lm_logits_duration_1[:, 2, self.discrete_value_ids[6]].view(-1, 1),
        ), 1)
        duration_val_1 = self.s(duration_val_1)

        duration_val_1 = torch.bmm(duration_val_1.view(-1, 1, 7), constant_vec).view(-1, 1)

        # We use the lm_logits for start time queries
        decoder_outputs = (lm_logits_original,) + (lm_logits_start,) + (lm_logits_duration_1,) + (lm_logits_duration_2,)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        regular_lm_loss = loss_fct(lm_logits_original.view(-1, lm_logits_original.size(-1)), lm_labels_original.view(-1))

        loss = regular_lm_loss

        # compute the inference for end time queries
        end_logic_loss = -duration_val_1 + dist
        decoder_outputs = (end_logic_loss, ) + decoder_outputs
        end_logic_loss = torch.cat((end_logic_loss, -end_logic_loss), 1)
        end_logic_loss = self.s(end_logic_loss)
        end_logic_loss = loss_fct(end_logic_loss, end_point_label.view(-1))

        loss += end_logic_loss

        decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if len(past) < 2:
            encoder_outputs, decoder_past_key_value_states = past, None
        else:
            encoder_outputs, decoder_past_key_value_states = past[0], past[1]

        return {
            "decoder_input_ids": input_ids,
            "decoder_past_key_value_states": decoder_past_key_value_states,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if len(past) < 2:
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()
        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return past + (reordered_decoder_past,)