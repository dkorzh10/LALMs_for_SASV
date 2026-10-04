from typing import Dict, Any, List, Union, Optional
import torch
import logging
from .base import Model
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import get_peft_model, TaskType
from .peft_utils import make_lora_config

class QwenAudioModel(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        qwen_cfg = config.get("additional_kwargs", {}).get("qwen_audio", {})
        model_path = qwen_cfg.get("model_path", "")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        
        # Determine dtype
        self.torch_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.torch_dtype = torch.bfloat16
            
        # Load model
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if not torch.distributed.is_initialized() else None,
            local_files_only=True
        )

        # Fix UserWarning: `pad_token_id` should be positive but got -1 (required for batch generation with padding)
        tok = self.processor.tokenizer
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None or pad_id < 0:
            pad_id = getattr(tok, "eos_token_id", None)
        if pad_id is not None and pad_id >= 0:
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                self.model.generation_config.pad_token_id = pad_id
        
        # LoRA setup
        lora_cfg = config.get("lora", {})
        ckpt_path = config.get("ckpt", "")
        
        if lora_cfg.get("enabled", False):
            peft_config = make_lora_config(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_cfg.get("rank", 8),
                lora_alpha=lora_cfg.get("alpha", 32),
                lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_rslora=True,
            )
            self.model = get_peft_model(self.model, peft_config)
            print(f"LoRA adapters initialized with rank={lora_cfg.get('rank', 8)}")
        
        # Load checkpoint if provided (AFTER LoRA setup so structure matches)
        if ckpt_path:
            import os
            print(f"Attempting to load checkpoint from: {ckpt_path}")
            
            if os.path.isdir(ckpt_path):
                # PEFT directory format (adapter_config.json, adapter_model.bin)
                if lora_cfg.get("enabled", False):
                    from peft import PeftModel
                    print(f"Loading PEFT adapter from directory: {ckpt_path}")
                    self.model = PeftModel.from_pretrained(self.model, ckpt_path)
                    print("PEFT adapter loaded successfully.")
                else:
                    raise ValueError(f"Checkpoint is a directory but LoRA is not enabled")
                    
            elif os.path.isfile(ckpt_path) and ckpt_path.endswith('.pt'):
                # State dict .pt file format
                print(f"Loading checkpoint from .pt file: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                
                # Extract model state dict (handle both wrapped and direct formats)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print(f"Checkpoint contains: epoch={checkpoint.get('epoch', 'unknown')}, "
                          f"val_accuracy={checkpoint.get('val_accuracy', 'unknown')}")
                else:
                    state_dict = checkpoint
                
                # Clean up state dict keys
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                
                # Sample a few keys to detect the prefix pattern
                sample_keys = list(state_dict.keys())[:5]
                print(f"Sample checkpoint keys: {sample_keys[:2]}")
                
                # Detect and strip wrapper prefixes
                for k, v in state_dict.items():
                    name = k
                    
                    # Strip 'module.' prefix (from DDP)
                    if name.startswith('module.'):
                        name = name.replace('module.', '', 1)
                    
                    # Strip 'model.' prefix (from wrapper model class)
                    # This happens when the checkpoint saves self.model.state_dict()
                    # where self.model is a wrapper containing the actual model
                    if name.startswith('model.'):
                        name = name.replace('model.', '', 1)
                    
                    new_state_dict[name] = v
                
                # Show what the keys look like after cleaning
                cleaned_sample = list(new_state_dict.keys())[:2]
                print(f"Cleaned keys: {cleaned_sample}")
                
                # Load state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                
                print(f"Checkpoint loaded from .pt file.")
                if missing_keys:
                    print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                
                # Sanity check: verify some LoRA weights were loaded
                if lora_cfg.get("enabled", False):
                    lora_weights_found = any('lora' in k.lower() for k in new_state_dict.keys())
                    if lora_weights_found:
                        print("  ✓ LoRA weights found and loaded in checkpoint")
                        # Count how many were actually loaded vs missing
                        lora_loaded = sum(1 for k in new_state_dict.keys() if 'lora' in k.lower() and k not in missing_keys)
                        lora_total = sum(1 for k in new_state_dict.keys() if 'lora' in k.lower())
                        print(f"  ✓ LoRA weights loaded: {lora_loaded}/{lora_total}")
                    else:
                        print("  ⚠ WARNING: No LoRA weights found in checkpoint! Model may not be properly loaded.")
            else:
                raise ValueError(f"Checkpoint path must be either a directory or .pt file: {ckpt_path}")
        
        # Ensure audio_tower is in the same dtype as the rest of the model
        # Must be done AFTER PEFT wrapping, accessing via base_model for PEFT models
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        if hasattr(base_model, 'audio_tower'):
            base_model.audio_tower = base_model.audio_tower.to(self.torch_dtype)
            print(f"Audio tower converted to {self.torch_dtype}")
        if hasattr(base_model, 'multi_modal_projector'):
            base_model.multi_modal_projector = base_model.multi_modal_projector.to(self.torch_dtype)
            print(f"Multi-modal projector converted to {self.torch_dtype}")
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("Gradient checkpointing enabled for Qwen Audio model")

    def get_tokenizer(self):
        return getattr(self.processor, "tokenizer", None) if hasattr(self, "processor") else None

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        # Qwen2Audio expects specific input format
        # samples['input_ids'] currently contains the prompt tokens (potentially padded)
        # samples['text'] contains the target text
        
        target_texts = [t + "<|im_end|>" + self.processor.tokenizer.eos_token for t in samples["text"]]
        
        # Use prompts from samples if available (for SASV format with multiple audio tokens)
        # Otherwise reconstruct the default prompt
        prompts = samples.get("prompts", None)
        
        full_texts = []
        for i in range(len(samples["text"])):
            if prompts and i < len(prompts):
                # Use provided prompt (already has correct format and multiple <|AUDIO|> tokens if needed)
                prompt = prompts[i]
            else:
                # Reconstruct the default prompt (single audio)
                prompt = f"<|im_start|>user\n<|AUDIO|>Analyze this audio for antispoofing.<|im_end|>\n<|im_start|>assistant\n"
            full_texts.append(prompt + target_texts[i])
            
        # Re-process with the full text to get correct input_ids and labels
        # We still use the input_features and feature_attention_mask from samples
        inputs = self.processor(
            text=full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False # Already have them in full_texts
        ).to(self.model.device)
        
        # Create labels; mask prompt tokens.
        labels = inputs.input_ids.clone()
        prompt_placeholder = "<|im_start|>assistant\n"
        for i in range(len(full_texts)):
            prompt_only = full_texts[i].split(prompt_placeholder)[0] + prompt_placeholder
            prompt_tokens = self.processor.tokenizer(prompt_only, add_special_tokens=False).input_ids
            labels[i, :len(prompt_tokens)] = -100
            
        # Also mask padding
        labels[inputs.input_ids == self.processor.tokenizer.pad_token_id] = -100
        
        model_inputs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
        }
        
        if samples.get("input_features") is not None:
            model_inputs["input_features"] = samples["input_features"].to(self.model.device, dtype=self.torch_dtype)
        
        if "feature_attention_mask" in samples and samples["feature_attention_mask"] is not None:
            model_inputs["feature_attention_mask"] = samples["feature_attention_mask"].to(self.model.device)
        
        outputs = self.model(**model_inputs)
        return {"loss": outputs.loss}

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any], prompts: Optional[List[str]] = None, return_outputs: bool = False) -> Union[List[str], Any]:
        # Implementation for generation
        # To be robust against padding, we reconstruct and re-process the prompt
        
        # Use prompts from samples if available (for SASV format with multiple audio tokens)
        full_prompts = []
        if prompts is not None:
            # If explicit prompts provided, use them (already formatted by caller)
            full_prompts = prompts
        elif "prompts" in samples and samples["prompts"]:
            # Use prompts from samples (already have correct format)
            full_prompts = samples["prompts"]
        else:
            # Reconstruct default prompt
            for i in range(len(samples["audio_ids"])):
                prompt = f"<|im_start|>user\n<|AUDIO|>Analyze this audio for antispoofing.<|im_end|>\n<|im_start|>assistant\n"
                full_prompts.append(prompt)
                
        inputs = self.processor(
            text=full_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        model_inputs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }
        
        if samples.get("input_features") is not None:
            model_inputs["input_features"] = samples["input_features"].to(self.model.device, dtype=self.torch_dtype)
            
        if "feature_attention_mask" in samples and samples["feature_attention_mask"] is not None:
            model_inputs["feature_attention_mask"] = samples.get("feature_attention_mask").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=generate_cfg.get("max_new_tokens", 128),
            do_sample=generate_cfg.get("do_sample", False),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
        )
        
        # Decode only the generated part
        input_ids_len = inputs.input_ids.shape[1]
        generated_ids_only = generated_ids[:, input_ids_len:]
        
        texts = self.processor.batch_decode(generated_ids_only, skip_special_tokens=True)
        
        if return_outputs:
            logits = self.compute_logits_for_completions(samples, generated_ids_only)
            return texts, generated_ids_only, logits
        return texts

    def compute_logits_for_completions(self, samples: Dict[str, Any], completion_ids: torch.Tensor) -> torch.Tensor:
        # Re-run model with completion_ids appended to input_ids.
        # Use the same prompts as in the batch so that the number of <|AUDIO|> placeholders
        # matches input_features (e.g. SASV has 2 audios per sample -> 2 placeholders per prompt).
        completion_texts = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        prompts = samples.get("prompts", None)
        full_texts = []
        for i in range(len(completion_texts)):
            if prompts and i < len(prompts):
                prompt = prompts[i]
            else:
                prompt = f"<|im_start|>user\n<|AUDIO|>Analyze this audio for antispoofing.<|im_end|>\n<|im_start|>assistant\n"
            full_texts.append(prompt + completion_texts[i])
            
        inputs = self.processor(
            text=full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        model_inputs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }
        
        if samples.get("input_features") is not None:
            model_inputs["input_features"] = samples["input_features"].to(self.model.device, dtype=self.torch_dtype)
            
        if "feature_attention_mask" in samples and samples["feature_attention_mask"] is not None:
            model_inputs["feature_attention_mask"] = samples.get("feature_attention_mask").to(self.model.device)
        
        outputs = self.model(**model_inputs)
        
        # Completion logits start after the prompt tokens.
        prompt_placeholder = "<|im_start|>assistant\n"
        logits_list = []
        for i in range(len(full_texts)):
            prompt_only = full_texts[i].split(prompt_placeholder)[0] + prompt_placeholder
            prompt_tokens = self.processor.tokenizer(prompt_only, add_special_tokens=False).input_ids
            # outputs.logits[i] predicts token at index i+1
            # Logit at len(prompt_tokens)-1 predicts first completion token
            sample_logits = outputs.logits[i, len(prompt_tokens)-1 : len(prompt_tokens)-1 + completion_ids.shape[1], :]
            logits_list.append(sample_logits)
            
        return torch.stack(logits_list)

    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict[str, Any]) -> List[str]:
        inputs = self.processor(text=prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=generate_cfg.get("max_new_tokens", 128),
            do_sample=generate_cfg.get("do_sample", False),
        )
        return self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)