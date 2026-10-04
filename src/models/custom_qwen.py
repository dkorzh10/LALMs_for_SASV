from .peft_utils import make_lora_config
class CustomQwenAudioModel(Model):
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

        # Fix UserWarning: pad_token_id
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

        # Load checkpoint if provided (AFTER LoRA setup)
        if ckpt_path:
            print(f"Attempting to load checkpoint from: {ckpt_path}")

            if os.path.isdir(ckpt_path):
                if lora_cfg.get("enabled", False):
                    print(f"Loading PEFT adapter from directory: {ckpt_path}")
                    self.model = PeftModel.from_pretrained(self.model, ckpt_path)
                    print("PEFT adapter loaded successfully.")
                else:
                    raise ValueError(f"Checkpoint is a directory but LoRA is not enabled")

            elif os.path.isfile(ckpt_path) and ckpt_path.endswith('.pt'):
                print(f"Loading checkpoint from .pt file: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')

                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print(f"Checkpoint contains: epoch={checkpoint.get('epoch', 'unknown')}, "
                          f"val_accuracy={checkpoint.get('val_accuracy', 'unknown')}")
                else:
                    state_dict = checkpoint

                new_state_dict = OrderedDict()
                sample_keys = list(state_dict.keys())[:5]
                print(f"Sample checkpoint keys: {sample_keys[:2]}")

                for k, v in state_dict.items():
                    name = k
                    if name.startswith('module.'):
                        name = name.replace('module.', '', 1)
                    if name.startswith('model.'):
                        name = name.replace('model.', '', 1)
                    new_state_dict[name] = v

                cleaned_sample = list(new_state_dict.keys())[:2]
                print(f"Cleaned keys: {cleaned_sample}")

                missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

                print(f"Checkpoint loaded from .pt file.")
                if missing_keys:
                    print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

                if lora_cfg.get("enabled", False):
                    lora_weights_found = any('lora' in k.lower() for k in new_state_dict.keys())
                    if lora_weights_found:
                        print("  ✓ LoRA weights found and loaded in checkpoint")
                        lora_loaded = sum(1 for k in new_state_dict.keys() if 'lora' in k.lower() and k not in missing_keys)
                        lora_total = sum(1 for k in new_state_dict.keys() if 'lora' in k.lower())
                        print(f"  ✓ LoRA weights loaded: {lora_loaded}/{lora_total}")
                    else:
                        print("  ⚠ WARNING: No LoRA weights found in checkpoint!")
            else:
                raise ValueError(f"Checkpoint path must be either a directory or .pt file: {ckpt_path}")

        # Ensure audio_tower and projector in correct dtype
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        if hasattr(base_model, 'audio_tower'):
            base_model.audio_tower = base_model.audio_tower.to(self.torch_dtype)
            print(f"Audio tower converted to {self.torch_dtype}")
        if hasattr(base_model, 'multi_modal_projector'):
            base_model.multi_modal_projector = base_model.multi_modal_projector.to(self.torch_dtype)
            print(f"Multi-modal projector converted to {self.torch_dtype}")

        # Custom loss (create once, reuse in forward)
        self.loss_fct = BioAntispoofingLoss(
            self.processor.tokenizer
        )

        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("Gradient checkpointing enabled for Qwen Audio model")

    def get_tokenizer(self):
        return getattr(self.processor, "tokenizer", None) if hasattr(self, "processor") else None

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        target_texts = [t + "<|im_end|>" + self.processor.tokenizer.eos_token for t in samples["text"]]
        prompts = samples.get("prompts", None)

        full_texts = []
        for i in range(len(samples["text"])):
            if prompts and i < len(prompts):
                prompt = prompts[i]
            else:
                prompt = f"<|im_start|>user\n<|AUDIO|>Analyze this audio for antispoofing.<|im_end|>\n<|im_start|>assistant\n"
            full_texts.append(prompt + target_texts[i])

        inputs = self.processor(
            text=full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        # Create labels
        labels = inputs.input_ids.clone()
        prompt_placeholder = "<|im_start|>assistant\n"
        for i in range(len(full_texts)):
            prompt_only = full_texts[i].split(prompt_placeholder)[0] + prompt_placeholder
            prompt_tokens = self.processor.tokenizer(prompt_only, add_special_tokens=False).input_ids
            labels[i, :len(prompt_tokens)] = -100

        labels[inputs.input_ids == self.processor.tokenizer.pad_token_id] = -100

        has_audio = samples.get("input_features") is not None

        model_inputs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }

        if has_audio:
            model_inputs["input_features"] = samples["input_features"].to(self.model.device, dtype=self.torch_dtype)

        if "feature_attention_mask" in samples and samples["feature_attention_mask"] is not None:
            model_inputs["feature_attention_mask"] = samples["feature_attention_mask"].to(self.model.device)

        # Forward without labels so we get logits in merged sequence space
        outputs = self.model(**model_inputs)

        if has_audio:
            # With audio, Qwen2Audio merges audio into the sequence; logits have merged length.
            # Build expanded labels to match and use custom weighted loss.
            base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
            # Config may use audio_token_id (newer) or audio_token_index (older transformers)
            audio_token_id = getattr(self.model.config, "audio_token_id", None) or getattr(
                self.model.config, "audio_token_index", None
            )
            if audio_token_id is None:
                audio_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
            ignore_index = getattr(self.model.config, "ignore_index", -100)
            expanded_labels = _expand_labels_for_merged_sequence(
                labels=labels.to(self.model.device),
                input_ids=inputs.input_ids.to(self.model.device),
                feature_attention_mask=samples["feature_attention_mask"].to(self.model.device),
                audio_tower=base.audio_tower,
                audio_token_id=audio_token_id,
                merged_seq_len=outputs.logits.shape[1],
                device=self.model.device,
                ignore_index=ignore_index,
            )
            loss = self.loss_fct(outputs.logits, expanded_labels)
        else:
            loss = self.loss_fct(outputs.logits, labels)

        return {"loss": loss}

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any], prompts: Optional[List[str]] = None, return_outputs: bool = False) -> Union[List[str], Any]:
        full_prompts = []
        if prompts is not None:
            full_prompts = prompts
        elif "prompts" in samples and samples["prompts"]:
            full_prompts = samples["prompts"]
        else:
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

        gen_kwargs = dict(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated = self.model.generate(**gen_kwargs)
        
        first_token_logits = generated.scores[0]
        
        tokenizer = self.processor.tokenizer
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        gen_ids = tokenizer.encode("gen", add_special_tokens=False)
        
        yes_id = yes_ids[-1] if yes_ids else None
        no_id = no_ids[-1] if no_ids else None
        gen_id = gen_ids[-1] if gen_ids else None
        
        results = []
        for batch_idx in range(first_token_logits.shape[0]):
            probs = {}
            
            if yes_id is not None:
                probs["yes"] = first_token_logits[batch_idx, yes_id].item()
            if no_id is not None:
                probs["no"] = first_token_logits[batch_idx, no_id].item()
            if gen_id is not None:
                probs["gen"] = first_token_logits[batch_idx, gen_id].item()
            
            if probs:
                result = max(probs, key=probs.get)
            else:
                token_id = generated.sequences[batch_idx, inputs.input_ids.shape[1]:].item()
                result = tokenizer.decode(token_id, skip_special_tokens=True).strip().lower()
                result = result.split()[0] if result else "unknown"
            
            results.append(result)
        
        if return_outputs:
            input_ids_len = inputs.input_ids.shape[1]
            generated_ids_only = generated.sequences[:, input_ids_len:]
            return results, generated_ids_only, first_token_logits
        
        return results

    def compute_logits_for_completions(self, samples: Dict[str, Any], completion_ids: torch.Tensor, scores=None) -> torch.Tensor:
        device = self.model.device

        if scores is not None:
            logits = torch.stack(scores, dim=1).to(device) 

            T_comp = completion_ids.size(1)
            if logits.size(1) != T_comp:
                logits = logits[:, -T_comp:, :]

            return logits 
        
        prompts = samples.get("prompts", None)

        full_prompts = []
        if prompts is not None:
            full_prompts = prompts
        else:
            for _ in range(completion_ids.size(0)):
                prompt = (
                    "<|im_start|>user\n"
                    "<|AUDIO|>Analyze this audio for antispoofing.<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                full_prompts.append(prompt)

        prompt_inputs = self.processor(
            text=full_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        input_ids = torch.cat([prompt_inputs.input_ids, completion_ids.to(device)], dim=1)
        attention_mask = torch.cat(
            [
                prompt_inputs.attention_mask,
                torch.ones_like(completion_ids, device=device),
            ],
            dim=1,
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if samples.get("input_features") is not None:
            model_inputs["input_features"] = samples["input_features"].to(device, dtype=self.torch_dtype)
        if "feature_attention_mask" in samples and samples["feature_attention_mask"] is not None:
            model_inputs["feature_attention_mask"] = samples["feature_attention_mask"].to(device)

        outputs = self.model(**model_inputs)

        T_comp = completion_ids.size(1)
        logits_completion = outputs.logits[:, -T_comp:, :] 

        return logits_completion

    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict[str, Any]) -> List[str]:
        inputs = self.processor(text=prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=generate_cfg.get("max_new_tokens", 128),
            do_sample=generate_cfg.get("do_sample", False),
        )
        return self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)