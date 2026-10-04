"""
Shared utilities for confidence extraction from model logits.

This module provides functions to extract Real/Fake probabilities from model logits,
supporting multi-token words (e.g., SALMON's "Fake" tokenized as ["F", "ake"]).
"""

import re
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def get_tokenizer(model):
    """
    Get tokenizer from model.
    
    Args:
        model: The unwrapped model instance
        
    Returns:
        Tokenizer instance or None if not found
    """
    if hasattr(model, 'processor'):
        return model.processor.tokenizer
    if hasattr(model, 'tokenizer'):
        return model.tokenizer
    if hasattr(model, 'model') and hasattr(model.model, 'llama_tokenizer'):
        return model.model.llama_tokenizer
    return None


def get_token_ids(tokenizer, cache: Optional[Dict] = None, task_type: str = "antispoofing") -> tuple:
    """
    Get token ID lists for answer classes.
    
    For antispoofing: returns (real_token_ids, fake_token_ids)
    For SASV: returns (yes_token_ids, no_token_ids, gen_token_ids)
    
    Returns lists to support multi-token words (e.g., SALMON: Fake -> [F, ake]).
    Uses cache dictionary if provided to avoid repeated tokenization.
    
    Args:
        tokenizer: Tokenizer instance
        cache: Optional dict to cache token IDs
        task_type: "antispoofing" or "sasv"
        
    Returns:
        For antispoofing: Tuple of (real_token_ids, fake_token_ids) as lists
        For SASV: Tuple of (yes_token_ids, no_token_ids, gen_token_ids) as lists
    """
    if task_type == "sasv":
        if cache is not None:
            if '_yes_token_ids' not in cache or cache['_yes_token_ids'] is None:
                cache['_yes_token_ids'] = tokenizer.encode("yes", add_special_tokens=False)
                cache['_no_token_ids'] = tokenizer.encode("no", add_special_tokens=False)
                cache['_gen_token_ids'] = tokenizer.encode("gen", add_special_tokens=False)
            return cache['_yes_token_ids'], cache['_no_token_ids'], cache['_gen_token_ids']
        else:
            yes_ids = tokenizer.encode("yes", add_special_tokens=False)
            no_ids = tokenizer.encode("no", add_special_tokens=False)
            gen_ids = tokenizer.encode("gen", add_special_tokens=False)
            return yes_ids, no_ids, gen_ids
    else:
        # Antispoofing format
        if cache is not None:
            if '_real_token_ids' not in cache or cache['_real_token_ids'] is None:
                cache['_real_token_ids'] = tokenizer.encode("Real", add_special_tokens=False)
                cache['_fake_token_ids'] = tokenizer.encode("Fake", add_special_tokens=False)
            return cache['_real_token_ids'], cache['_fake_token_ids']
        else:
            real_ids = tokenizer.encode("Real", add_special_tokens=False)
            fake_ids = tokenizer.encode("Fake", add_special_tokens=False)
            return real_ids, fake_ids


def find_answer_position(text: str, tokenizer, token_ids: Optional[List[int]] = None, 
                         real_ids: Optional[List[int]] = None, fake_ids: Optional[List[int]] = None,
                         yes_ids: Optional[List[int]] = None, no_ids: Optional[List[int]] = None, 
                         gen_ids: Optional[List[int]] = None, task_type: str = "antispoofing") -> int:
    """
    Find position of answer token in generated sequence.
    
    If token_ids provided, searches directly in token sequence (more reliable).
    Otherwise falls back to re-tokenizing the decoded text.
    
    Args:
        text: Decoded text string
        tokenizer: Tokenizer instance
        token_ids: Optional list of token IDs for the generated sequence
        real_ids: Optional pre-computed token IDs for "Real" (antispoofing)
        fake_ids: Optional pre-computed token IDs for "Fake" (antispoofing)
        yes_ids: Optional pre-computed token IDs for "yes" (SASV)
        no_ids: Optional pre-computed token IDs for "no" (SASV)
        gen_ids: Optional pre-computed token IDs for "gen" (SASV)
        task_type: "antispoofing" or "sasv"
        
    Returns:
        Position index of the answer token
    """
    # SASV reasoning: the answer word follows the rightmost "<answer>" tag.
    # We locate it by re-tokenizing the decoded prefix up to and including the
    # tag. This is reliable because it tokenizes a prefix of the *same* string
    # that produced ``token_ids`` (so boundaries line up), whereas matching the
    # standalone ``encode("<answer>")`` / ``encode("gen")`` fails: SentencePiece
    # adds a leading space ("▁<", "▁gen") that is absent mid-sequence.
    if task_type == "sasv":
        m = None
        for m in re.finditer(r"<answer>", text):
            pass  # keep the last match
        if m is not None:
            prefix = text[: m.end()]
            return len(tokenizer.encode(prefix, add_special_tokens=False))
        # Fallback: rightmost standalone yes/no/gen id-subsequence.
        if token_ids is not None and yes_ids is not None and no_ids is not None and gen_ids is not None:
            n = len(token_ids)
            max_len = max(len(yes_ids), len(no_ids), len(gen_ids))
            for i in range(n - max_len, -1, -1):
                for ids in (yes_ids, no_ids, gen_ids):
                    Li = len(ids)
                    if i + Li <= n and token_ids[i : i + Li] == ids:
                        return i

    # If we have the actual token IDs, search for answer tokens directly
    if token_ids is not None:
        if real_ids is not None and fake_ids is not None:
            # Search for Real or Fake token sequence in the generated tokens
            for i in range(len(token_ids) - max(len(real_ids), len(fake_ids)) + 1):
                # Check if Real tokens match at position i
                if token_ids[i:i+len(real_ids)] == real_ids:
                    return i
                # Check if Fake tokens match at position i
                if token_ids[i:i+len(fake_ids)] == fake_ids:
                    return i
            
            # Fallback: search backwards from end (answer is usually at the end)
            for i in range(len(token_ids) - 1, -1, -1):
                if i + len(real_ids) <= len(token_ids) and token_ids[i:i+len(real_ids)] == real_ids:
                    return i
                if i + len(fake_ids) <= len(token_ids) and token_ids[i:i+len(fake_ids)] == fake_ids:
                    return i
    
    # Fallback: re-tokenize decoded text (less reliable with PAD tokens)
    # Reasoning format: after <answer>
    match = re.search(r"<answer>", text)
    if match:
        prefix = text[:match.end()]
        return len(tokenizer.encode(prefix, add_special_tokens=False))
    
    # Hard-label format: after "Final Answer: "
    match = re.search(r"Final Answer:\s*", text)
    if match:
        prefix = text[:match.end()]
        return len(tokenizer.encode(prefix, add_special_tokens=False))
    
    return 0  # First token as fallback


def extract_confidences(pred_texts: List[str], logits: torch.Tensor, 
                       tokenizer, real_ids: Optional[List[int]] = None, fake_ids: Optional[List[int]] = None,
                       yes_ids: Optional[List[int]] = None, no_ids: Optional[List[int]] = None,
                       gen_ids: Optional[List[int]] = None,
                       token_ids: Optional[torch.Tensor] = None,
                       extract_answer_fn=None, task_type: str = "antispoofing") -> List[Dict[str, float]]:
    """
    Extract answer class probabilities from logits.
    
    Supports both antispoofing (Real/Fake) and SASV (yes/no/gen) formats.
    For multi-token words (e.g., SALMON 'Fake' -> F, ake), uses product of token probabilities
    in log-space for numerical stability.
    
    Args:
        pred_texts: List of decoded prediction texts
        logits: Logits tensor [batch_size, seq_len, vocab_size]
        tokenizer: Tokenizer instance
        real_ids: Token IDs for "Real" (antispoofing)
        fake_ids: Token IDs for "Fake" (antispoofing)
        yes_ids: Token IDs for "yes" (SASV)
        no_ids: Token IDs for "no" (SASV)
        gen_ids: Token IDs for "gen" (SASV)
        token_ids: Optional token IDs tensor [batch_size, seq_len] for more reliable position finding
        extract_answer_fn: Optional function to extract answer from text (e.g., _extract_answer method)
        task_type: "antispoofing" or "sasv"
    
    Returns:
        List of dicts with probabilities and confidence
        For antispoofing: real_prob, fake_prob, confidence
        For SASV: yes_prob, no_prob, gen_prob, confidence
    """
    if logits is None:
        if task_type == "sasv":
            return [{"yes_prob": 0.0, "no_prob": 0.0, "gen_prob": 0.0, "confidence": 0.33} for _ in pred_texts]
        else:
            return [{"real_prob": 0.0, "fake_prob": 0.0, "confidence": 0.5} for _ in pred_texts]

    # SASVSalmonModel (and similar): answer head logits [batch, 3] for yes / no / gen — not LM [B, T, V]
    if task_type == "sasv" and logits.dim() == 2 and logits.shape[-1] == 3:
        probs = F.softmax(logits.float(), dim=-1)
        confidences = []
        for i, text in enumerate(pred_texts):
            yes_prob = probs[i, 0].item()
            no_prob = probs[i, 1].item()
            gen_prob = probs[i, 2].item()
            if extract_answer_fn is not None:
                pred_answer = extract_answer_fn(text).lower()
            else:
                text_lower = text.lower()
                if re.search(r"\byes\b", text_lower):
                    pred_answer = "yes"
                elif re.search(r"\bno\b", text_lower) and not re.search(r"\b(gen|generated|spoof)\b", text_lower):
                    pred_answer = "no"
                elif re.search(r"\bgen\b", text_lower) or re.search(r"\b(generated|spoof)\b", text_lower):
                    pred_answer = "gen"
                else:
                    pred_answer = ""
            if pred_answer == "yes":
                confidence = yes_prob
            elif pred_answer == "no":
                confidence = no_prob
            elif pred_answer == "gen":
                confidence = gen_prob
            else:
                confidence = max(yes_prob, no_prob, gen_prob)
            confidences.append({
                "yes_prob": yes_prob,
                "no_prob": no_prob,
                "gen_prob": gen_prob,
                "confidence": confidence,
            })
        return confidences

    if tokenizer is None:
        if task_type == "sasv":
            return [{"yes_prob": 0.0, "no_prob": 0.0, "gen_prob": 0.0, "confidence": 0.33} for _ in pred_texts]
        else:
            return [{"real_prob": 0.0, "fake_prob": 0.0, "confidence": 0.5} for _ in pred_texts]

    confidences = []
    seq_len = logits.shape[1]
    vocab_size = logits.shape[2]

    # For SASV reasoning, the answer word is emitted right after the "<answer>"
    # tag, so its in-context tokenization (e.g. "gen" after ">") differs from the
    # standalone encode("gen") == "▁gen". Anchor on the tag and score the answer
    # using the in-context ids to avoid frozen/misaligned probabilities.
    answer_anchor_ids = None
    yes_ctx_ids, no_ctx_ids, gen_ctx_ids = yes_ids, no_ids, gen_ids
    if task_type == "sasv" and tokenizer is not None:
        try:
            answer_anchor_ids = tokenizer.encode("<answer>", add_special_tokens=False)
            n_anchor = len(answer_anchor_ids)

            def _ctx_ids(word: str, fallback):
                full = tokenizer.encode("<answer>" + word, add_special_tokens=False)
                if len(full) > n_anchor and full[:n_anchor] == answer_anchor_ids:
                    return full[n_anchor:]
                return fallback

            yes_ctx_ids = _ctx_ids("yes", yes_ids)
            no_ctx_ids = _ctx_ids("no", no_ids)
            gen_ctx_ids = _ctx_ids("gen", gen_ids)
        except Exception:
            answer_anchor_ids = None
            yes_ctx_ids, no_ctx_ids, gen_ctx_ids = yes_ids, no_ids, gen_ids

    def seq_logprob(ids: list, start: int, batch_idx: int) -> float:
        """Compute log-probability of token sequence for numerical stability."""
        if start + len(ids) > seq_len:
            return -float('inf')
        logp = 0.0
        for k, tid in enumerate(ids):
            pos = start + k
            if pos >= seq_len or tid >= vocab_size:
                return -float('inf')
            token_logits = logits[batch_idx, pos].float()
            logprobs = F.log_softmax(token_logits, dim=-1)
            logp += logprobs[tid].item()
        return logp

    for i, text in enumerate(pred_texts):
        # Use token IDs if available for more reliable position finding
        token_list = token_ids[i].tolist() if token_ids is not None else None
        
        if task_type == "sasv" and yes_ids is not None and no_ids is not None and gen_ids is not None:
            # SASV format: three classes
            answer_pos = find_answer_position(text, tokenizer, token_list, 
                                             yes_ids=yes_ids, no_ids=no_ids, gen_ids=gen_ids,
                                             task_type="sasv")
            
            yes_prob = 0.0
            no_prob = 0.0
            gen_prob = 0.0

            if answer_pos < seq_len:
                log_yes = seq_logprob(yes_ctx_ids, answer_pos, i)
                log_no = seq_logprob(no_ctx_ids, answer_pos, i)
                log_gen = seq_logprob(gen_ctx_ids, answer_pos, i)

                # Normalize probabilities to sum to 1 using logsumexp for numerical stability
                log_total = torch.logsumexp(torch.tensor([log_yes, log_no, log_gen]), dim=0)
                yes_prob = torch.exp(log_yes - log_total).item()
                no_prob = torch.exp(log_no - log_total).item()
                gen_prob = torch.exp(log_gen - log_total).item()
                
                # Confidence = P(predicted class)
                if extract_answer_fn is not None:
                    pred_answer = extract_answer_fn(text).lower()
                else:
                    text_lower = text.lower()
                    if re.search(r'\byes\b', text_lower):
                        pred_answer = "yes"
                    elif re.search(r'\bno\b', text_lower) and not re.search(r'\b(gen|generated|spoof)\b', text_lower):
                        pred_answer = "no"
                    elif re.search(r'\bgen\b', text_lower) or re.search(r'\b(generated|spoof)\b', text_lower):
                        pred_answer = "gen"
                    else:
                        pred_answer = ""
                
                if pred_answer == "yes":
                    confidence = yes_prob
                elif pred_answer == "no":
                    confidence = no_prob
                elif pred_answer == "gen":
                    confidence = gen_prob
                else:
                    # If we can't determine prediction, use the highest probability
                    confidence = max(yes_prob, no_prob, gen_prob)
            else:
                yes_prob = no_prob = gen_prob = 1.0 / 3.0
                confidence = 1.0 / 3.0

            confidences.append({
                "yes_prob": yes_prob,
                "no_prob": no_prob,
                "gen_prob": gen_prob,
                "confidence": confidence
            })
        else:
            # Antispoofing format: two classes
            answer_pos = find_answer_position(text, tokenizer, token_list, real_ids, fake_ids, task_type="antispoofing")

            real_prob = 0.0
            fake_prob = 0.0

            if answer_pos < seq_len:
                log_real = seq_logprob(real_ids, answer_pos, i)
                log_fake = seq_logprob(fake_ids, answer_pos, i)

                # Normalize probabilities to sum to 1 using logsumexp for numerical stability
                log_total = torch.logsumexp(torch.tensor([log_real, log_fake]), dim=0)
                real_prob = torch.exp(log_real - log_total).item()
                fake_prob = torch.exp(log_fake - log_total).item()
                
                # Confidence = P(predicted class) for proper EER computation
                # Determine which class was predicted by checking the text
                if extract_answer_fn is not None:
                    pred_answer = extract_answer_fn(text).lower()
                else:
                    # Simple fallback: check if "fake" or "real" appears in text
                    text_lower = text.lower()
                    if "fake" in text_lower:
                        pred_answer = "fake"
                    elif "real" in text_lower:
                        pred_answer = "real"
                    else:
                        pred_answer = ""
                
                if pred_answer == "fake":
                    confidence = fake_prob
                elif pred_answer == "real":
                    confidence = real_prob
                else:
                    # If we can't determine prediction, use the higher probability
                    confidence = max(real_prob, fake_prob)
            else:
                real_prob = fake_prob = 0.5
                confidence = 0.5

            confidences.append({
                "real_prob": real_prob,
                "fake_prob": fake_prob,
                "confidence": confidence
            })

    return confidences
