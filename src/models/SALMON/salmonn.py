# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

import logging
import json
import contextlib
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList
from peft import TaskType, get_peft_model
from ..peft_utils import make_lora_config

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM
from .modeling_whisper import WhisperModel
from .utils import StoppingCriteriaSub


# BEATs is optional - only needed if beats_path is specified
try:
    from .beats.BEATs import BEATsConfig, BEATs
    BEATS_AVAILABLE = True
except (ImportError, Exception):
    BEATS_AVAILABLE = False
    BEATsConfig = None
    BEATs = None
    logging.warning("BEATs not available - training without BEATs encoder")

from transformers import WavLMModel


def checkpoint_has_lora(state_dict):
    return any("lora_" in k for k in state_dict.keys())



class SALMONN(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=None):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise detect hardware support
        enable_autocast = self.device != torch.device("cpu")

        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        if enable_autocast:
            return torch.amp.autocast("cuda", dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        whisper_unfreeze_last_n_layers=0,  # Unfreeze last N layers (0 = all frozen)
        whisper_unfreeze_attention_only=False,  # If True, only unfreeze attention in those layers
        beats_path="",
        freeze_beats=True,

        wavlm_path="",
        freeze_wavlm=True,

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        wrap_collator_prompts=True,
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        torch_dtype=torch.float16, # Default to float16
        
        # Class weighting for imbalanced datasets
        use_class_weights=True,  # Enable weighted loss by default
        class_weights=None,  # Will be computed from dataset if None
        bonafide_weight_multiplier=1.0,  # Additional multiplier for bonafide class (e.g., 2.0 to double its weight)
        # SASV (Spoofing-Aware Speaker Verification): explicit weights for yes/no/gen
        # Higher weight = more penalty when wrong (biometric: yes/no). Lower = less penalty (spoof: gen)
        sasv_class_weights=None,  # e.g. {"yes": 2.0, "no": 2.0, "gen": 0.5}
    ):
        super().__init__()

        self.whisper_path = whisper_path
        self.beats_path = beats_path
        self.wavlm_path = wavlm_path
        self.freeze_wavlm = freeze_wavlm
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.wrap_collator_prompts = wrap_collator_prompts
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        
        # Store class weighting parameters
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        self.bonafide_weight_multiplier = bonafide_weight_multiplier
        self.sasv_class_weights = sasv_class_weights  # {"yes": w, "no": w, "gen": w} for SASV
        self.class_weight_tensor = None  # Will be set later when moved to device
        self.sasv_class_weights = sasv_class_weights

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False, local_files_only=True)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch_dtype,
                load_in_8bit=True,
                device_map={"": device_8bit},
                local_files_only=True,
                vocab_size=len(self.llama_tokenizer)
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
                # vocab_size=len(self.llama_tokenizer)
            )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        if self.lora:
            self.peft_config = make_lora_config(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ],
                use_rslora=True,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')

        # Use config from local llama path if possible, or a local bert config
        # For SALMONN, this is used for the Q-Former.
        # We should avoid downloading "bert-base-uncased"
        try:
            whisper_model = WhisperModel.from_pretrained(whisper_path, local_files_only=True)
        except AttributeError:
            # Exported encoders may ship safetensors without HF metadata headers.
            whisper_model = WhisperModel.from_pretrained(
                whisper_path, local_files_only=True, use_safetensors=False
            )
        self.speech_encoder = whisper_model.encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)

        # Handle Whisper freezing with selective unfreezing
        if freeze_whisper:
            # First, freeze all parameters
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            
            # Selectively unfreeze last N layers if requested
            if whisper_unfreeze_last_n_layers > 0:
                total_layers = len(self.speech_encoder.layers)
                start_layer = max(0, total_layers - whisper_unfreeze_last_n_layers)
                
                trainable_params = 0
                for i in range(start_layer, total_layers):
                    if whisper_unfreeze_attention_only:
                        # Only unfreeze attention layers
                        for param in self.speech_encoder.layers[i].self_attn.parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()
                    else:
                        # Unfreeze entire layer
                        for param in self.speech_encoder.layers[i].parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()
                
                if whisper_unfreeze_attention_only:
                    logging.info(f"Whisper: frozen except attention in last {whisper_unfreeze_last_n_layers} layers ({trainable_params/1e6:.1f}M params)")
                else:
                    logging.info(f"Whisper: frozen except last {whisper_unfreeze_last_n_layers} layers ({trainable_params/1e6:.1f}M params)")
            else:
                logging.info("Whisper: fully frozen")
        else:
            total_params = sum(p.numel() for p in self.speech_encoder.parameters())
            logging.info(f"Whisper: fully trainable ({total_params/1e6:.1f}M params)")
        
        if self.beats_path:
            if not BEATS_AVAILABLE:
                logging.warning("BEATs path specified but BEATs module not available. Skipping BEATs.")
                self.beats_path = ""  # Disable BEATs
            else:
                logging.info("Loading BEATs Model")
                try:
                    beats_ckpt = torch.load(self.beats_path, map_location='cpu')
                    beats_cfg = BEATsConfig(beats_ckpt['cfg'])
                    self.beats = BEATs(beats_cfg)
                    self.beats.load_state_dict(beats_ckpt['model'])
                    self.ln_beats = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
                    if freeze_beats:
                        for name, param in self.beats.named_parameters():
                            param.requires_grad = False
                        self.beats.eval()
                        logging.info("freeze BEATs")
                except Exception as e:
                    logging.warning(f"Failed to load BEATs: {e}. Continuing without BEATs.")
                    self.beats_path = ""  # Disable BEATs

        if self.wavlm_path:
            logging.info("Loading WavLM Model")
            try:
                self.wavlm = WavLMModel.from_pretrained(self.wavlm_path, local_files_only=True)
                self.ln_wavlm = nn.LayerNorm(self.wavlm.config.hidden_size)
                if freeze_wavlm:
                    for name, param in self.wavlm.named_parameters():
                        param.requires_grad = False
                    self.wavlm.eval()
                    logging.info("freeze WavLM")
            except Exception as e:
                logging.warning(f"Failed to load WavLM: {e}. Continuing without WavLM.")
                self.wavlm_path = ""  # Disable WavLM

        if self.use_speech_Qformer:

            Qformer_speech_width = 0
            if self.whisper_path:
                Qformer_speech_width += self.speech_encoder.config.d_model
            if self.beats_path:
                Qformer_speech_width += self.beats.cfg.encoder_embed_dim
            if self.wavlm_path:
                Qformer_speech_width += self.wavlm.config.hidden_size

            self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                num_query_token=num_speech_query_token, speech_width=Qformer_speech_width
            )

            # if self.beats_path:
            #     self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            #         num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
            #     )
            # else:
            #     self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            #         num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
            #     )

            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                state = torch.load(speech_llama_proj_model, map_location="cpu")
                if 'model' in state:
                    state = state['model']

                self.speech_llama_proj.load_state_dict(state, strict=True)

            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            print('!!!!!!!!loading prompts from:', prompt_path)
            print('!!!!!!!!loading prompts from:', prompt_path)
            print('!!!!!!!!loading prompts from:', prompt_path)
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")
            print('prompt dict:', self.prompt_dict)
            print('prompt dict:', self.prompt_dict)
            print('prompt dict:', self.prompt_dict)

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None, wavlm_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                # speech_embeds = self.ln_speech(speech_embeds)
                # if audio_embeds is not None:
                #     audio_embeds = self.ln_audio(audio_embeds)
                #     if audio_embeds.size(1) < speech_embeds.size(1):
                #         audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                #     elif audio_embeds.size(1) > speech_embeds.size(1):
                #         speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                #     speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                speech_embeds = self.ln_speech(speech_embeds)

                feature_streams = [speech_embeds]
                target_len = speech_embeds.size(1)

                if audio_embeds is not None:
                    audio_embeds = self.ln_beats(audio_embeds)
                    feature_streams.append(audio_embeds)
                    target_len = max(target_len, audio_embeds.size(1))

                if wavlm_embeds is not None:
                    wavlm_embeds = self.ln_wavlm(wavlm_embeds)
                    feature_streams.append(wavlm_embeds)
                    target_len = max(target_len, wavlm_embeds.size(1))

                if len(feature_streams) > 1:
                    feature_streams = [
                        F.pad(stream, (0, 0, 0, target_len - stream.size(1)))
                        if stream.size(1) < target_len else stream
                        for stream in feature_streams
                    ]
                    speech_embeds = torch.cat(feature_streams, dim=-1)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def _coerce_raw_wav_for_beats(self, raw_wav, audio_padding_mask=None, device=None):
        """Accept list[np.ndarray], np.ndarray, or [B, T] tensor (as in SALMONN_SDR collate)."""
        if raw_wav is None:
            return None, audio_padding_mask

        if device is None and hasattr(self, "beats"):
            device = next(self.beats.parameters()).device

        if isinstance(raw_wav, list):
            tensors = []
            for w in raw_wav:
                if isinstance(w, np.ndarray):
                    tensors.append(torch.from_numpy(w).float().flatten())
                elif isinstance(w, torch.Tensor):
                    tensors.append(w.float().flatten())
                else:
                    tensors.append(torch.tensor(w, dtype=torch.float32).flatten())
            lengths = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
            raw_wav = pad_sequence(tensors, batch_first=True, padding_value=0.0)
            if audio_padding_mask is None:
                audio_padding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        elif isinstance(raw_wav, np.ndarray):
            raw_wav = torch.from_numpy(raw_wav).float()
            if raw_wav.dim() == 1:
                raw_wav = raw_wav.unsqueeze(0)
        elif isinstance(raw_wav, torch.Tensor) and raw_wav.dim() == 1:
            raw_wav = raw_wav.unsqueeze(0)

        if device is not None:
            raw_wav = raw_wav.to(device)
            if audio_padding_mask is not None:
                audio_padding_mask = audio_padding_mask.to(device)
        return raw_wav, audio_padding_mask

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

            audio_embeds = None
            wavlm_embeds = None
            if (self.beats_path or self.wavlm_path) and raw_wav is None:
                raise ValueError("raw_wav is required when BEATs or WavLM is enabled")
            if (self.beats_path or self.wavlm_path) and raw_wav is not None:
                raw_wav, audio_padding_mask = self._coerce_raw_wav_for_beats(
                    raw_wav, audio_padding_mask, device=spectrogram.device
                )

            if self.beats_path:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            if self.wavlm_path:
                wavlm_attention_mask = None
                if audio_padding_mask is not None:
                    wavlm_attention_mask = (~audio_padding_mask).long()
                wavlm_outputs = self.wavlm(
                    input_values=raw_wav,
                    attention_mask=wavlm_attention_mask,
                    return_dict=True,
                )
                wavlm_embeds = wavlm_outputs.last_hidden_state

        return self._encode_auditory_feature(
            speech_embeds,
            audio_embeds=audio_embeds,
            wavlm_embeds=wavlm_embeds,
        )

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat(
                    [
                        p_before_tokens.attention_mask.expand(batch_size, -1),
                        atts,
                        p_after_tokens.attention_mask.expand(batch_size, -1),
                    ],
                    dim=1,
                )
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def set_sasv_class_weights(self, weights_dict=None):
        """
        Set explicit SASV class weights for yes/no/gen.
        Higher weight = more penalty when model predicts wrong (biometric: yes, no).
        Lower weight = less penalty (spoof: gen).
        
        Args:
            weights_dict: {"yes": float, "no": float, "gen": float}, e.g. {"yes": 2.0, "no": 2.0, "gen": 0.5}

        Returns:
            Applied mapping {"yes": float, "no": float, "gen": float} or None if disabled/empty.
        """
        if not self.use_class_weights:
            logging.info("SASV class weighting is disabled (use_class_weights=False)")
            return

        source_weights = self.sasv_class_weights if weights_dict is None else weights_dict
        if not source_weights:
            logging.info("No SASV class weights provided; skip weighted CE setup")
            return

        normalized = {}
        alias = {
            "yes": "yes",
            "true": "yes",
            "no": "no",
            "false": "no",
            "gen": "gen",
        }
        for raw_key, raw_weight in source_weights.items():
            key = alias.get(str(raw_key).strip().lower())
            if key is None:
                logging.warning(
                    "Ignoring unknown SASV class weight key %r (supported: yes/no/gen or true/false)",
                    raw_key,
                )
                continue
            try:
                normalized[key] = float(raw_weight)
            except (TypeError, ValueError):
                logging.warning(
                    "Ignoring invalid SASV class weight value for key %r: %r",
                    raw_key,
                    raw_weight,
                )

        if not normalized:
            logging.warning("No valid SASV class weights after normalization; skip weighted CE setup")
            return

        applied = {
            "yes": normalized.get("yes", 1.0),
            "no": normalized.get("no", 1.0),
            "gen": normalized.get("gen", 1.0),
        }

        vocab_size = len(self.llama_tokenizer)
        class_weight_tensor = torch.ones(vocab_size)

        for word, weight in applied.items():
            tokens = self.llama_tokenizer(
                word + self.end_sym,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0]
            for token_id in tokens.tolist():
                class_weight_tensor[token_id] = weight
        
        device = next(self.parameters()).device
        class_weight_tensor = class_weight_tensor.to(device)
        self.class_weight_tensor = class_weight_tensor
        
        if self.lora:
            self.llama_model.base_model.model.config.class_weight_tensor = class_weight_tensor
        else:
            self.llama_model.config.class_weight_tensor = class_weight_tensor

        self.applied_sasv_class_weights = applied
        logging.info(
            "SASV class weights applied (raw=%s -> normalized=%s)",
            source_weights,
            applied,
        )
        return applied

    def set_class_weights_from_dataset(self, dataset):
        """
        Compute class weights from dataset to handle class imbalance.
        Uses inverse frequency weighting: weight = n_samples / (n_classes * n_samples_per_class)
        If sasv_class_weights is provided, uses those instead (biometric vs spoof weighting).
        
        Args:
            dataset: SALMONNDataset instance with annotation field
        """
        if not self.use_class_weights:
            logging.info("Class weighting disabled")
            return
        
        # Prefer explicit SASV weights if provided
        if self.sasv_class_weights is not None:
            self.set_sasv_class_weights()
            return
            
        # Count occurrences of each target text
        from collections import Counter
        target_counts = Counter()
        
        for item in dataset.annotation:
            target_counts[item['text']] += 1
        
        total_samples = len(dataset.annotation)
        n_classes = len(target_counts)
        
        logging.info(f"Computing class weights from {total_samples} samples, {n_classes} classes:")
        for text, count in target_counts.items():
            logging.info(f"  '{text}': {count} samples ({count/total_samples*100:.1f}%)")
        
        # Tokenize each target to get token IDs
        # We'll weight tokens based on which answer they belong to
        token_weights = {}
        
        for text, count in target_counts.items():
            # Calculate weight for this class
            weight = total_samples / (n_classes * count)
            

            # # Apply multiplier for bonafide class if configured                              ДЛЯ СПУФИНГА
            # if 'bonafide' in text.lower() and hasattr(self, 'bonafide_weight_multiplier'):
            #     weight *= self.bonafide_weight_multiplier
            #     logging.info(f"  Applying bonafide weight multiplier {self.bonafide_weight_multiplier}x to '{text}'")
            
            # Tokenize to get the token IDs for this answer
            tokens = self.llama_tokenizer(
                text + self.end_sym,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0]
            
            for token_id in tokens.tolist():
                if token_id not in token_weights:
                    token_weights[token_id] = []
                token_weights[token_id].append(weight)
        
        # Average weights for tokens that appear in multiple classes
        for token_id in token_weights:
            token_weights[token_id] = sum(token_weights[token_id]) / len(token_weights[token_id])
        
        # Create weight tensor for all vocab (default weight = 1.0)
        vocab_size = len(self.llama_tokenizer)
        class_weight_tensor = torch.ones(vocab_size)
        
        for token_id, weight in token_weights.items():
            class_weight_tensor[token_id] = weight
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        class_weight_tensor = class_weight_tensor.to(device)
        
        self.class_weight_tensor = class_weight_tensor
        
        # Store in LLaMA model config so it's accessible during forward pass
        if self.lora:
            self.llama_model.base_model.model.config.class_weight_tensor = class_weight_tensor
        else:
            self.llama_model.config.class_weight_tensor = class_weight_tensor
        
        # Log the weights for the answer tokens
        logging.info("Token weights for answer classes:")
        for text in target_counts.keys():
            tokens = self.llama_tokenizer(text + self.end_sym, add_special_tokens=False).input_ids
            weights = [class_weight_tensor[tid].item() for tid in tokens]
            logging.info(f"  '{text}': tokens {tokens[:5]}... weights {[f'{w:.3f}' for w in weights[:5]]}...")

    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                print('!!!!!!!!choosing random prompt:', self.prompt_dict[samples["task"][0]])
                print('!!!!!!!!choosing random prompt:', self.prompt_dict[samples["task"][0]])
                print('!!!!!!!!choosing random prompt:', self.prompt_dict[samples["task"][0]])
                prompt = random.choice(self.prompt_dict[samples["task"][0]])
        
        print('!!!!!!!!prompt:', prompt)
        print('!!!!!!!!prompt:', prompt)
        print('!!!!!!!!prompt:', prompt)

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def compute_logits(self, samples, completion_ids, prompts=None):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # Explicit prompts (e.g. SASV batch instruction prompts) take priority so the
        # scoring input matches training. Otherwise fall back to the prompt_dict logic.
        prompt = prompts
        use_multi_for_prompts = isinstance(prompts, (list, tuple))
        if prompt is None and self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if prompt is not None:
            multi = use_multi_for_prompts or (self.prompt_dict and self.multi_prompt)
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=multi)

        # prepare inputs for LLM using completion_ids
        to_regress_ids = completion_ids.to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_ids)
        
        # Create attention mask for completions (assuming 0 is pad, or checking against pad_token_id)
        # However, completion_ids might not have padding if we generated them carefully, or we should pass mask.
        # For now, assume simple attention mask (ones) if not padded, or derive from ids.
        if self.llama_tokenizer.pad_token_id is not None:
             to_regress_mask = (to_regress_ids != self.llama_tokenizer.pad_token_id).long()
        else:
             to_regress_mask = torch.ones_like(to_regress_ids)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_ids.dtype,
            device=to_regress_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_mask], dim=1)

        # calulate logits
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            # Logits for the completions (skipping prompt/speech)
            # Input was [BOS, SPEECH, COMPLETION]
            # Output logits will be same length.
            # We want logits corresponding to the completion tokens.
            # The logits at index i predict token at i+1.
            # So logits for completion[0] comes from inputs[last_speech_idx].
            
            # Length of prefix (BOS + SPEECH)
            prefix_len = 1 + speech_embeds.shape[1]
            
            # The logits corresponding to generating the completion tokens:
            # We want P(completion | prefix).
            # outputs.logits shape: [B, SeqLen, Vocab]
            # We want logits starting from the end of speech.
            # logits[:, prefix_len-1 : -1, :] 
            # If we want to evaluate the probability of completion_ids:
            # logits[i] predicts input[i+1].
            # Input: [BOS, S1...Sn, C1...Cm]
            # Logits: [L_BOS, L_S1...L_Sn, L_C1...L_Cm]
            # L_Sn is prediction for C1.
            # L_Cm-1 is prediction for Cm.
            # So we take logits from index (prefix_len - 1) to (prefix_len + completion_len - 1).
            
            # Slice to return only the logits corresponding to completion tokens.
            # Logits at position i predict token at position i+1.
            # Input: [BOS, S1...Sn, C1...Cm]
            # We want logits that predict C1...Cm, which are at positions prefix_len-1 to prefix_len+m-2.
            # Actually for consistency with completion_ids shape, we want m logits.
            completion_len = completion_ids.shape[1]
            sliced_logits = outputs.logits[:, prefix_len-1 : prefix_len-1+completion_len, :]
            
            return sliced_logits

    def generate(self, samples, generate_cfg, prompts=None, return_generation_info=False):
        batch_size = samples["spectrogram"].shape[0]
        num_return_sequences = generate_cfg.get("num_return_sequences", 1)

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        # Encode audio once per sample; expand encoded state for k generations.
        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        if num_return_sequences > 1:
            speech_embeds = speech_embeds.repeat_interleave(num_return_sequences, dim=0)
            speech_atts = speech_atts.repeat_interleave(num_return_sequences, dim=0)
            if prompts is not None:
                prompts = [p for p in prompts for _ in range(num_return_sequences)]
            batch_size = speech_embeds.shape[0]

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        # Create dummy input_ids to satisfy generation loop tracking
        # Use pad_token_id for speech positions
        input_ids = torch.ones(
            (batch_size, embeds.shape[1]),
            dtype=torch.long,
            device=embeds.device
        ) * (self.llama_tokenizer.pad_token_id if self.llama_tokenizer.pad_token_id is not None else 0)
        # Set BOS at the beginning (matching bos_embeds)
        input_ids[:, 0] = self.llama_tokenizer.bos_token_id

        stop_words_ids = [torch.tensor([2]).to(self.device)]  
        # Also add [PAD] token to stopping criteria if it's not the same as EOS
        if self.llama_tokenizer.pad_token_id is not None and self.llama_tokenizer.pad_token_id != 2:
             stop_words_ids.append(torch.tensor([self.llama_tokenizer.pad_token_id]).to(self.device))
             
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        was_training = self.llama_model.training
        gc_was_enabled = getattr(self.llama_model, "is_gradient_checkpointing", False)
        if was_training:
            self.llama_model.eval()
            if gc_was_enabled and hasattr(self.llama_model, "gradient_checkpointing_disable"):
                self.llama_model.gradient_checkpointing_disable()
        
        # Workaround for newer transformers where generate method is not directly available
        # We need to access the generation method through the model hierarchy
        try:
            # Try direct generate (works with older transformers)
            if hasattr(self.llama_model, 'generate'):
                generation_model = self.llama_model
            # For LoRA models, try to access through base_model
            elif hasattr(self.llama_model, 'base_model'):
                # Import GenerationMixin and bind the method
                from transformers import GenerationMixin
                generation_model = self.llama_model.base_model.model
                # Dynamically add generate method if missing
                if not hasattr(generation_model, 'generate'):
                    # Bind GenerationMixin methods to the model instance
                    for method_name in ['generate', '_prepare_attention_mask_for_generation', 
                                       '_prepare_encoder_decoder_kwargs_for_generation',
                                       '_expand_inputs_for_generation']:
                        if hasattr(GenerationMixin, method_name):
                            method = getattr(GenerationMixin, method_name)
                            setattr(generation_model, method_name, method.__get__(generation_model))
            else:
                generation_model = self.llama_model
            
            # Fix UserWarning: `pad_token_id` should be positive but got -1 (required for batch generation with padding)
            if hasattr(generation_model, "generation_config"):
                if generation_model.generation_config.pad_token_id is None or generation_model.generation_config.pad_token_id < 0:
                    generation_model.generation_config.pad_token_id = self.llama_tokenizer.pad_token_id

            do_sample = bool(generate_cfg.get("do_sample", False))
            num_beams = int(generate_cfg.get("num_beams", 1 if do_sample else 4))
            if "use_cache" in generate_cfg:
                use_cache = bool(generate_cfg["use_cache"])
            else:
                use_cache = True
            input_length = embeds.shape[1]
            max_new_tokens_requested = int(generate_cfg.get("max_new_tokens", 200))
            max_position_embeddings = getattr(
                getattr(generation_model, "config", None), "max_position_embeddings", 4096
            )
            max_new_tokens = min(max_new_tokens_requested, max(1, max_position_embeddings - input_length))
            if max_new_tokens < max_new_tokens_requested:
                logging.warning(
                    "Generation capped to %d new tokens (requested %d) because input length %d + requested "
                    "would exceed model max_position_embeddings=%d.",
                    max_new_tokens, max_new_tokens_requested, input_length, max_position_embeddings,
                )
            generation_kwargs = {
                "input_ids": input_ids,
                "inputs_embeds": embeds,
                "max_new_tokens": max_new_tokens,
                "stopping_criteria": stopping_criteria,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "min_length": generate_cfg.get("min_length", 1),
                "repetition_penalty": generate_cfg.get("repetition_penalty", 1.0),
                "length_penalty": generate_cfg.get("length_penalty", 1.0),
                "attention_mask": attns,
                "pad_token_id": self.llama_tokenizer.pad_token_id,
                "eos_token_id": self.llama_tokenizer.eos_token_id,  # Ensure EOS is respected
                "use_cache": use_cache,
            }
            if do_sample:
                generation_kwargs["temperature"] = generate_cfg.get("temperature", 1.0)
                generation_kwargs["top_p"] = generate_cfg.get("top_p", 0.9)

            with self.maybe_autocast():
                outputs = generation_model.generate(**generation_kwargs)
        except AttributeError as e:
            logging.error(f"Generate method not available: {e}")
            logging.error("This is likely due to transformers version incompatibility")
            raise RuntimeError(
                "The model's generate() method is not available. "
                "This may be due to transformers>=4.50 removing generate from PreTrainedModel. "
                "Please use transformers<4.50 or update the model code."
            )
        finally:
            if was_training:
                if gc_was_enabled and hasattr(self.llama_model, "gradient_checkpointing_enable"):
                    self.llama_model.gradient_checkpointing_enable()
                self.llama_model.train()
        # ``outputs`` is the full sequence (prefix input_ids + new tokens). Prefix positions
        # are pad (speech) except BOS; decoding the whole row yields "<s>[pad]..." garbage.
        prefix_len = int(input_ids.shape[1])
        out_len = int(outputs.shape[1])
        if out_len > prefix_len:
            gen_ids = outputs[:, prefix_len:]
        else:
            gen_ids = None
        if gen_ids is not None and gen_ids.numel() > 0:
            text = self.llama_tokenizer.batch_decode(
                gen_ids, add_special_tokens=False, skip_special_tokens=True
            )
        else:
            text = [""] * int(outputs.shape[0])
        text = [str(t).strip() for t in text]

        if return_generation_info:
            pad_token_id = self.llama_tokenizer.pad_token_id
            if gen_ids is not None and gen_ids.numel() > 0:
                if pad_token_id is None:
                    generated_lens = torch.full(
                        (gen_ids.shape[0],),
                        gen_ids.shape[1],
                        dtype=torch.long,
                        device=gen_ids.device,
                    )
                else:
                    generated_lens = (gen_ids != pad_token_id).long().sum(dim=1)
            else:
                generated_lens = torch.zeros(batch_size, dtype=torch.long, device=embeds.device)

            prefix_lens = attns.long().sum(dim=1)
            total_lens = prefix_lens + generated_lens
            context_window = getattr(getattr(generation_model, "config", None), "max_position_embeddings", None)
            generation_info = {
                "max_new_tokens": int(generate_cfg.get("max_new_tokens", 200)),
                "context_window": int(context_window) if context_window is not None else None,
                "padded_prefix_len": int(prefix_len),
                "prefix_lens": prefix_lens.detach().cpu().tolist(),
                "generated_lens": generated_lens.detach().cpu().tolist(),
                "total_lens": total_lens.detach().cpu().tolist(),
            }
            return text, generation_info

        return text


    @classmethod
    def from_config(cls, config):
        import logging
        import torch
        from peft import set_peft_model_state_dict

        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        whisper_unfreeze_last_n_layers = config.get("whisper_unfreeze_last_n_layers", 0)
        whisper_unfreeze_attention_only = config.get("whisper_unfreeze_attention_only", False)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)
        wavlm_path = config.get("wavlm_path", "")
        freeze_wavlm = config.get("freeze_wavlm", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        wrap_collator_prompts = config.get("wrap_collator_prompts", True)
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        torch_dtype = config.get("torch_dtype", torch.float16)
        if isinstance(torch_dtype, str):
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[torch_dtype]

        use_class_weights = config.get("use_class_weights", True)
        class_weights = config.get("class_weights", None)
        bonafide_weight_multiplier = config.get("bonafide_weight_multiplier", 1.0)
        sasv_class_weights = config.get("sasv_class_weights", None)  # {"yes": 2.0, "no": 2.0, "gen": 0.5}

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            whisper_unfreeze_last_n_layers=whisper_unfreeze_last_n_layers,
            whisper_unfreeze_attention_only=whisper_unfreeze_attention_only,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            wavlm_path=wavlm_path,
            freeze_wavlm=freeze_wavlm,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            wrap_collator_prompts=wrap_collator_prompts,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            torch_dtype=torch_dtype,
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            bonafide_weight_multiplier=bonafide_weight_multiplier,
            sasv_class_weights=sasv_class_weights,
        )


        ckpt_path = config.get("ckpt", "")
        debug = config.get("debug", False) # if True, print the model architecture and parameters
        new_lora = config.get("new_lora", False)
        if debug:
            print("Model architecture:")
            print(model)
            print("Model parameters:")
            # for name, param in model.named_parameters():
            #     print(f"{name}: {param.shape}")

        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["model"]
            if debug:
                print("Checkpoint state:")
                for k, v in state.items():
                    print(f"{k}: {tuple(v.shape)}")
            # If checkpoint was saved from SalmonModel (unified_training wrapper), keys have "model." prefix
            if state and any(k.startswith("model.") for k in state.keys()):
                state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
            if new_lora and state:
                state = {k: v for k, v in state.items() if "lora" not in k}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logging.warning(f"Missing keys: {missing}")
            if unexpected:
                logging.warning(f"Unexpected keys: {unexpected}")

        return model