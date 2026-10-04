import json
import os
import random
import warnings
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional, Tuple, Union
import math
import torch.nn.functional as F
import torchaudio
from dataclasses import dataclass
from scipy import signal
import copy

# -------------------------
#  My Utils
# -------------------------
def _pad_raw_wav_batch(
    waveforms: List[Union[np.ndarray, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length waveforms to [B, T] and build a BEATs padding mask."""
    tensors: List[torch.Tensor] = []
    for w in waveforms:
        if isinstance(w, np.ndarray):
            t = torch.from_numpy(w).float().flatten()
        elif isinstance(w, torch.Tensor):
            t = w.float().flatten()
        else:
            t = torch.tensor(w, dtype=torch.float32).flatten()
        tensors.append(t)
    lengths = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
    padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
    padding_mask = torch.arange(padded.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
    return padded, padding_mask


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav, sr

def _read_audio_any(path: str) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    if not os.path.exists(path):
        return None, None
    try:
        wav, sr = torchaudio.load(path)
        wav = wav.float()
        return wav, sr
    except Exception:
        try:
            wav_np, sr = sf.read(path)
            wav = torch.from_numpy(wav_np)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.transpose(0, 1)
            return wav.float(), int(sr)
        except Exception:
            return None, None

def _power(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (x ** 2).mean().clamp_min(eps)

def _snr_mix(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    cp = _power(clean)
    np_ = _power(noise)
    snr = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(cp / (snr * np_))
    return clean + noise * scale

def _random_crop(wav: torch.Tensor, num_samples: int, mode: str = "random") -> torch.Tensor:
    T = wav.shape[-1]
    if T == num_samples:
        return wav
    if T < num_samples:
        return wav
    if mode == "center":
        start = (T - num_samples) // 2
    else:
        start = random.randint(0, T - num_samples)
    return wav[..., start:start + num_samples]

def _pad_to_len(wav: torch.Tensor, num_samples: int, pad_mode: str = "repeat") -> torch.Tensor:
    T = wav.shape[-1]
    if T >= num_samples:
        return wav[..., :num_samples]
    if pad_mode == "zero":
        pad = num_samples - T
        return F.pad(wav, (0, pad))
    if T == 0:
        return torch.zeros_like(wav[..., :1]).repeat(1, num_samples)
    reps = math.ceil(num_samples / T)
    rep = wav.repeat(1, reps)[..., :num_samples]
    return rep

def _fft_convolve_1d(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    T = x.shape[-1]
    L = h.shape[-1]
    n = 1 << (T + L - 1).bit_length()
    X = torch.fft.rfft(x, n=n)
    H = torch.fft.rfft(h, n=n)
    y = torch.fft.irfft(X * H, n=n)[..., :T]
    return y

def _rand_range(x1: float, x2: float, integer: bool = False) -> float:
    y = np.random.uniform(low=x1, high=x2, size=(1,))[0]
    if integer:
        y = int(y)
    return y

def _norm_wav(x: np.ndarray, always: bool = False) -> np.ndarray:
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x

def _gen_notch_coeffs(n_bands: int, min_f: float, max_f: float, min_bw: float, max_bw: float,
                     min_coeff: int, max_coeff: int, min_g: float, max_g: float, fs: int) -> np.ndarray:
    b = np.array([1.0])
    for i in range(n_bands):
        fc = _rand_range(min_f, max_f, False)
        bw = _rand_range(min_bw, max_bw, False)
        c = _rand_range(min_coeff, max_coeff, True)
        
        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs), b)
    G = _rand_range(min_g, max_g, False)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b

def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2):int(y.shape[0] - N / 2)]
    return y

# ------------------------------------------------------------------------------------------------


@dataclass
class VadConfig:
    enabled: bool = True
    trigger_level: float = 7.0
    search_time: float = 0.2
    allowed_gap: float = 0.25
    pre_trigger_time: float = 0.0
    boot_time: float = 0.15

def apply_sox_vad(wav: torch.Tensor, sr: int, cfg: VadConfig) -> torch.Tensor:
    """
    Returns speech-only waveform. If VAD fails/returns empty -> returns original wav.
    Requires torchaudio built with sox support.
    """
    if not cfg.enabled:
        return wav
    try:
        out = torchaudio.functional.vad(
            wav,
            sample_rate=sr,
            trigger_level=cfg.trigger_level,
            search_time=cfg.search_time,
            allowed_gap=cfg.allowed_gap,
            pre_trigger_time=cfg.pre_trigger_time,
            boot_time=cfg.boot_time,
        )
        if out.numel() < sr * 0.05:
            return wav
        return out
    except Exception:
        return wav

@dataclass
class AugConfig:
    enabled: bool = True
    p_noise: float = 0.0
    snr_db_min: float = 15.0
    snr_db_max: float = 25.0
    noise_paths: Optional[List[str]] = None
    p_rir: float = 0.0
    rir_paths: Optional[List[str]] = None
    rir_max_len_sec: float = 2.0
    p_fir_filter: float = 0.5
    lowpass_freq_min: float = 3400.0
    lowpass_freq_max: float = 7500.0
    highpass_freq_min: float = 100.0
    highpass_freq_max: float = 400.0
    # RawBoost parameters
    p_rawboost: float = 0.5
    # LnL_convolutive_noise parameters
    n_bands: int = 5
    min_f: int = 20
    max_f: int = 8000
    min_bw: int = 100
    max_bw: int = 1000
    min_coeff: int = 10
    max_coeff: int = 100
    min_g: int = 0
    max_g: int = 0
    min_bias_lin_nonlin: int = 5
    max_bias_lin_nonlin: int = 20
    n_f: int = 5
    # ISD_additive_noise parameters
    p_impulse: int = 10
    g_sd: int = 2

class AudioAugmenter:
    def __init__(self, target_sr: int, max_len_sec: float, min_len_sec: float,
                 pad_mode: str, crop_mode: str,
                 vad_cfg: VadConfig, aug_cfg: AugConfig, 
                 enroll_max_len_sec: Optional[float] = None, query_max_len_sec: Optional[float] = None):
        self.target_sr = target_sr
        self.max_len_sec = max_len_sec
        self.min_len_sec = min_len_sec
        self.pad_mode = pad_mode
        self.crop_mode = crop_mode
        self.vad_cfg = vad_cfg
        self.aug_cfg = aug_cfg

        self.max_len = int(round(max_len_sec * target_sr))
        self.min_len = int(round(min_len_sec * target_sr))
        
        self.enroll_max_len = int(round(enroll_max_len_sec * target_sr)) if enroll_max_len_sec else self.max_len
        self.query_max_len = int(round(query_max_len_sec * target_sr)) if query_max_len_sec else self.max_len
        
        self.enroll_max_len_sec = enroll_max_len_sec or max_len_sec
        self.query_max_len_sec = query_max_len_sec or max_len_sec

    @staticmethod
    def _load_list(txt_path: Optional[str]) -> List[str]:
        if not txt_path:
            return []
        if not os.path.exists(txt_path):
            return []
        with open(txt_path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def maybe_add_noise(self, clean: torch.Tensor) -> torch.Tensor:
        if (not self.aug_cfg.enabled) or (not self.aug_cfg.noise_paths) or (random.random() > self.aug_cfg.p_noise):
            return clean
        noise_path = random.choice(self.aug_cfg.noise_paths)
        noise, nsr = _read_audio_any(noise_path)
        if noise is None:
            return clean
        noise = _to_mono(noise)
        noise, _ = _resample_if_needed(noise, nsr, self.target_sr)
        noise = _random_crop(noise, clean.shape[-1], mode="random")
        noise = _pad_to_len(noise, clean.shape[-1], pad_mode="repeat")
        snr_db = random.uniform(self.aug_cfg.snr_db_min, self.aug_cfg.snr_db_max)
        return _snr_mix(clean, noise, snr_db)

    def maybe_apply_rir(self, wav: torch.Tensor) -> torch.Tensor:
        # Temporarily disable RIR augmentation to avoid tensor dimension issues
        return wav

    def maybe_apply_fir_filter(self, wav: torch.Tensor) -> torch.Tensor:
        if (not self.aug_cfg.enabled) or (random.random() > self.aug_cfg.p_fir_filter):
            return wav
        filter_type = random.choice(['lowpass', 'highpass'])
        try:
            if filter_type == 'lowpass':
                freq = random.uniform(self.aug_cfg.lowpass_freq_min, self.aug_cfg.lowpass_freq_max)
                effects = [['sinc', '-n', '500', f'{freq:.0f}']]
                filtered_wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav, self.target_sr, effects=effects
                )
            else:
                freq = random.uniform(self.aug_cfg.highpass_freq_min, self.aug_cfg.highpass_freq_max)
                effects = [['sinc', '-n', '500', f'-{freq:.0f}']]
                filtered_wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav, self.target_sr, effects=effects
                )
            if filtered_wav.shape[-1] != wav.shape[-1]:
                if filtered_wav.shape[-1] > wav.shape[-1]:
                    filtered_wav = filtered_wav[..., :wav.shape[-1]]
                else:
                    filtered_wav = F.pad(filtered_wav, (0, wav.shape[-1] - filtered_wav.shape[-1]))
            return filtered_wav

        except Exception:
            return wav

    def _lnl_convolutive_noise(self, wav: torch.Tensor) -> torch.Tensor:
        wav_np = wav.squeeze().numpy()
        y = np.zeros_like(wav_np)
        for i in range(self.aug_cfg.n_f):
            if i == 1:
                min_g = self.aug_cfg.min_g - self.aug_cfg.min_bias_lin_nonlin
                max_g = self.aug_cfg.max_g - self.aug_cfg.max_bias_lin_nonlin
            else:
                min_g = self.aug_cfg.min_g
                max_g = self.aug_cfg.max_g
            
            b = _gen_notch_coeffs(
                self.aug_cfg.n_bands,
                self.aug_cfg.min_f,
                self.aug_cfg.max_f,
                self.aug_cfg.min_bw,
                self.aug_cfg.max_bw,
                self.aug_cfg.min_coeff,
                self.aug_cfg.max_coeff,
                min_g,
                max_g,
                self.target_sr
            )
            y = y + _filter_fir(np.power(wav_np, (i + 1)), b)
        y = y - np.mean(y)
        y = _norm_wav(y, False)
        return torch.from_numpy(y).unsqueeze(0)

    def _isd_additive_noise(self, wav: torch.Tensor) -> torch.Tensor:
        wav_np = wav.squeeze().numpy()
        beta = _rand_range(0, self.aug_cfg.p_impulse, False)
        
        y = copy.deepcopy(wav_np)
        x_len = wav_np.shape[0]
        n = int(x_len * (beta / 100))
        if n > 0:
            p = np.random.permutation(x_len)[:n]
            f_r = np.multiply(((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1))
            r = self.aug_cfg.g_sd * wav_np[p] * f_r
            y[p] = wav_np[p] + r
        y = _norm_wav(y, False)
        return torch.from_numpy(y).unsqueeze(0)

    def maybe_apply_rawboost(self, wav: torch.Tensor) -> torch.Tensor:
        # https://github.com/TakHemlata/RawBoost-antispoofing/tree/main
        if (not self.aug_cfg.enabled) or (random.random() > self.aug_cfg.p_rawboost):
            return wav
        try:
            wav = self._lnl_convolutive_noise(wav)
            wav = self._isd_additive_noise(wav)
            return wav
        except Exception:
            return wav

    def crop_and_pad(self, wav: torch.Tensor, audio_type: str = "default") -> torch.Tensor:
        T = wav.shape[-1]
        if audio_type == "enroll":
            max_len = self.enroll_max_len
        elif audio_type == "query":
            max_len = self.query_max_len
        else:
            max_len = self.max_len
        if T >= max_len:
            return _random_crop(wav, max_len, mode=self.crop_mode)
        return _pad_to_len(wav, max_len, pad_mode=self.pad_mode)

    def __call__(self, wav: torch.Tensor, sr: int, audio_type: str = "default") -> Tuple[torch.Tensor, int]:
        wav = _to_mono(wav)
        wav, sr = _resample_if_needed(wav, sr, self.target_sr)
        wav_vad = apply_sox_vad(wav, sr, self.vad_cfg)
        wav_use = wav_vad
        wav_use = self.crop_and_pad(wav_use, audio_type)

        wav_use = self.maybe_apply_rir(wav_use)
        wav_use = self.maybe_apply_rawboost(wav_use)
        wav_use = self.maybe_apply_fir_filter(wav_use)
        wav_use = self.maybe_add_noise(wav_use)

        wav_use = wav_use.clamp(-1.0, 1.0)
        return wav_use, sr


_EVAL_SPLITS = frozenset({"val", "dev", "test", "eval"})

def resolve_reasoning_text(item: Dict[str, Any], reasoning_version: str = "long") -> str:
    """Select long or short reasoning text from a dataset record."""
    version = (reasoning_version or "long").strip().lower()
    if version == "short":
        return (
            item.get("reasoning_short")
            or item.get("reasoning")
            or item.get("reasoning_original", "")
        )
    return item.get("reasoning_original") or item.get("reasoning", "")




def _resolve_audio_split(split: Optional[str], is_train: bool) -> str:
    if split is None:
        return "train" if is_train else "val"
    normalized = str(split).strip().lower()
    if normalized in _EVAL_SPLITS:
        return "val" if normalized in ("val", "dev") else "test"
    if normalized == "train":
        return "train"
    raise ValueError(f"Unknown audio split {split!r}; expected train, val, dev, or test")


class AudioDataset(Dataset):
    def __init__(self, data_path: str,
                 max_samples: Optional[int] = None,
                 target_sample_rate: int = 16000,
                 task_type: str = "hard_label",
                 samples_offset: int = 0,
                 audio_cfg: Optional[dict] = None,
                 is_train: bool = True,
                 split: Optional[str] = None,
                 reasoning_version: str = "long"):
        self.data_path = data_path
        self.target_sample_rate = target_sample_rate
        self.task_type = task_type
        self.reasoning_version = (reasoning_version or "long").strip().lower()
        self.samples: List[Dict[str, Any]] = []
        self.split = _resolve_audio_split(split, is_train)
        self.is_train = self.split == "train"
        
        self.samples_offset = samples_offset
        self._num_broken = 0

        self.audio_cfg = audio_cfg or {}
        self.target_sample_rate = self.audio_cfg.get("target_sr", target_sample_rate)

        vad = self.audio_cfg.get("vad", {})
        aug = self.audio_cfg.get("aug", {})
        crop = self.audio_cfg.get("crop", {})

        noise_list = []
        if aug.get("noise_paths_txt"):
            try:
                with open(aug["noise_paths_txt"], "r") as f:
                    noise_list = [ln.strip() for ln in f if ln.strip()]
            except FileNotFoundError:
                pass

        rir_list = []
        if aug.get("rir_paths_txt"):
            try:
                with open(aug["rir_paths_txt"], "r") as f:
                    rir_list = [ln.strip() for ln in f if ln.strip()]
            except FileNotFoundError:
                pass

        if self.is_train:
            enroll_max_len_sec = self.audio_cfg.get("train_enroll_max_len_sec")
            query_max_len_sec = self.audio_cfg.get("train_query_max_len_sec")
            default_max_len_sec = self.audio_cfg.get("train_max_len_sec", self.audio_cfg.get("max_len_sec", 8.0))
            crop_mode = str(crop.get("mode", "random"))
            aug_enabled = bool(aug.get("enabled", False))
        else:
            enroll_max_len_sec = self.audio_cfg.get("test_enroll_max_len_sec")
            query_max_len_sec = self.audio_cfg.get("test_query_max_len_sec")
            default_max_len_sec = self.audio_cfg.get("test_max_len_sec", self.audio_cfg.get("max_len_sec", 8.0))
            crop_mode = str(crop.get("test_mode", "center"))
            aug_enabled = False

        self.augmenter = AudioAugmenter(
            target_sr=self.target_sample_rate,
            max_len_sec=float(default_max_len_sec),
            min_len_sec=float(self.audio_cfg.get("min_len_sec", 1.0)),
            pad_mode=str(self.audio_cfg.get("pad_mode", "repeat")),
            crop_mode=crop_mode,

            vad_cfg=VadConfig(
                enabled=bool(vad.get("enabled", True)),
                trigger_level=float(vad.get("trigger_level", 7.0)),
                search_time=float(vad.get("search_time", 0.2)),
                allowed_gap=float(vad.get("allowed_gap", 0.25)),
                pre_trigger_time=float(vad.get("pre_trigger_time", 0.0)),
                boot_time=float(vad.get("boot_time", 0.15)),
            ),

            aug_cfg=AugConfig(
                enabled=aug_enabled,
                p_noise=float(aug.get("p_noise", 0.0)),
                snr_db_min=float(aug.get("snr_db", [15, 25])[0]),
                snr_db_max=float(aug.get("snr_db", [15, 25])[1]),
                noise_paths=noise_list,
                p_rir=float(aug.get("p_rir", 0.0)),
                rir_paths=rir_list,
                rir_max_len_sec=float(aug.get("rir_max_len_sec", 2.0)),
                p_fir_filter=float(aug.get("p_fir_filter", 0.5)),
                lowpass_freq_min=float(aug.get("lowpass_freq_min", 3400.0)),
                lowpass_freq_max=float(aug.get("lowpass_freq_max", 7500.0)),
                highpass_freq_min=float(aug.get("highpass_freq_min", 100.0)),
                highpass_freq_max=float(aug.get("highpass_freq_max", 400.0)),
                # RawBoost parameters
                p_rawboost=float(aug.get("p_rawboost", 0.5)),
                # LnL_convolutive_noise parameters
                n_bands=int(aug.get("n_bands", 5)),
                min_f=int(aug.get("min_f", 20)),
                max_f=int(aug.get("max_f", 8000)),
                min_bw=int(aug.get("min_bw", 100)),
                max_bw=int(aug.get("max_bw", 1000)),
                min_coeff=int(aug.get("min_coeff", 10)),
                max_coeff=int(aug.get("max_coeff", 100)),
                min_g=int(aug.get("min_g", 0)),
                max_g=int(aug.get("max_g", 0)),
                min_bias_lin_nonlin=int(aug.get("min_bias_lin_nonlin", 5)),
                max_bias_lin_nonlin=int(aug.get("max_bias_lin_nonlin", 20)),
                n_f=int(aug.get("n_f", 5)),
                # ISD_additive_noise parameters
                p_impulse=int(aug.get("p_impulse", 10)),
                g_sd=int(aug.get("g_sd", 2)),
            ),
            enroll_max_len_sec=enroll_max_len_sec,
            query_max_len_sec=query_max_len_sec,
        )
        self._load_data(max_samples)

    def _load_data(self, max_samples: Optional[int]):
        if self.data_path == "dummy":
            self.samples = [
                {
                    "audio_id": f"dummy_{i}",
                    "original_path": f"/tmp/dummy_audio_{i}.wav",
                    "is_bonafide": i % 2 == 0,
                    "reasons": {"strange_voice": True} if i % 2 != 0 else None,
                    "reasoning": "This sounds fake." if i % 2 != 0 else "This sounds real."
                }
                for i in range(100)
            ]
        else:
            try:
                with open(self.data_path, 'r') as f:
                    if self.data_path.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            self.samples = data
                        else:
                            print(f"Warning: Unexpected JSON format in {self.data_path}. Expecting list.")
                            self.samples = []
                    elif self.data_path.endswith('.jsonl'):
                        self.samples = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Warning: Dataset file {self.data_path} not found. Using empty dataset.")
                self.samples = []

        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load_audio(self, path: str, audio_type: str = "default"):
        wav, sr = _read_audio_any(path)
        if wav is None:
            return torch.zeros(1, 16000), 16000
        wav, sr = self.augmenter(wav, sr, audio_type)
        if wav.numel() == 0:
            return torch.zeros(1, 16000), 16000
        return wav, sr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        real_idx = self._num_broken + self.samples_offset + idx
        if real_idx >= len(self):
            real_idx %= len(self)
        item = self.samples[real_idx]
        
        # Check if this is SASV format (has reference_audios and query_audios)
        has_reference_audios = "reference_audios" in item and isinstance(item["reference_audios"], list)
        has_query_audios = "query_audios" in item and isinstance(item["query_audios"], list)
        
        if has_reference_audios and has_query_audios:
            reference_audios = []
            query_audios = []
            
            # Load reference audios (enroll samples)
            reference_speaker_ids = []
            for ref_audio in item["reference_audios"]:
                path = ref_audio.get("original_path") or ref_audio.get("path", "")
                waveform, sample_rate = self._load_audio(path, audio_type="enroll")
                if waveform is None and sample_rate is None:
                    self._num_broken += 1
                    return self.__getitem__(idx)
                reference_audios.append(waveform)
                reference_speaker_ids.append(ref_audio.get("speaker_id", ""))
            
            # Load query audios
            query_speaker_ids = []
            for query_audio in item["query_audios"]:
                path = query_audio.get("original_path") or query_audio.get("path", "")
                waveform, sample_rate = self._load_audio(path, audio_type="query")
                if waveform is None and sample_rate is None:
                    self._num_broken += 1
                    return self.__getitem__(idx)
                query_audios.append(waveform)
                query_speaker_ids.append(query_audio.get("speaker_id", ""))
            
            gt = item.get("gt", "").lower()
            if gt == "verified":
                answer = "yes"
            elif gt == "rejected":
                answer = "no"
            elif gt == "spoof":
                answer = "gen"
            else:
                answer = gt if gt in ["yes", "no", "gen"] else "no"
            
            # Determine target text based on task_type
            if self.task_type == "reasoning":
                reasoning = resolve_reasoning_text(item, self.reasoning_version)
                reasons = item.get("reasons", "")
                features = item.get("Acoustic features", "")
                target_text = (
                    f"<features>{features}</features>"
                    f"<think>{reasoning}</think>"
                    f"<reasons>{reasons}</reasons>"
                    f"<answer>{answer}</answer>"
                )
            else:
                reasoning = ""
                target_text = answer

            audio_id = item.get("task_id") or item.get("audio_id", f"sample_{real_idx}")

            return {
                "audio_id": audio_id,
                "reference_audios": reference_audios,
                "query_audios": query_audios,
                "reference_speaker_ids": reference_speaker_ids,
                "query_speaker_ids": query_speaker_ids,
                "gt": answer,
                "answer": answer,
                "task": "sasv",
                "text": target_text,
                "reasoning": reasoning,
            }
        else:
            # Original format: single audio per sample
            waveform, sample_rate = self._load_audio(item.get("original_path", ""))
            if waveform is None and sample_rate is None:
                self._num_broken += 1
                return self.__getitem__(idx)
            
            # Determine target text based on task_type
            is_bonafide = item.get("is_bonafide")
            reasoning = item.get("reasoning")
            reasons_dict = item.get("reasons", {})
            
            if self.task_type == "reasoning":
                think_content = reasoning if reasoning else ""
                
                reasons_list = []
                if isinstance(reasons_dict, dict):
                    reasons_list = [k.upper() for k, v in reasons_dict.items() if v]
                elif isinstance(reasons_dict, list):
                    reasons_list = [str(r).upper() for r in reasons_dict]
                elif isinstance(reasons_dict, str) and reasons_dict:
                    reasons_list = [reasons_dict.upper()]
                
                answer_content = "Real" if is_bonafide else "Fake"
                
                target_text = f"<think>{think_content}</think><reasons>{reasons_list}</reasons><answer>{answer_content}</answer>"
            else:
                target_text = "Real" if is_bonafide else "Fake"
                
            reasoning_out = reasoning if reasoning is not None else ""

            return {
                "audio_id": item.get("audio_id"),
                "raw_wav": waveform, 
                "original_path": item.get("original_path"),
                "is_bonafide": is_bonafide,
                "reasoning": reasoning_out,
                "task": "antispoofing", 
                "text": target_text
            }


def _format_prompt_for_model(prompt_template: str, model_name: str, reference_audios: Optional[List] = None, query_audios: Optional[List] = None) -> str:
    """Format a prompt template for the specific model's chat format."""
    if model_name in ("qwen_audio", "custom_qwen"):
        audio_token = "<|AUDIO|>"
        speech_token = "<|AUDIO|>"
    else:
        audio_token = ""
        # SALMONN prompt_wrap expects the exact placeholder token.
        speech_token = "<SpeechHere>"
    
    prompt_text = prompt_template.replace("<Audio>", audio_token)
    
    if reference_audios is not None and query_audios is not None:
        # SALMON-family models concatenate all waveforms into a single audio stream,
        # so prompt_wrap expects exactly one <SpeechHere> token.
        if model_name in ("salmon", "sasv_salmon", "new_salmon", "new_salmon_fusion"):
            ref_count = len(reference_audios)
            query_count = len(query_audios)
            stream_desc = (
                f"reference clips: {ref_count}, query clips: {query_count}, "
                f"concatenated audio stream: {speech_token}"
            )
            prompt_text = prompt_text.replace("<ReferenceAudios>", stream_desc)
            prompt_text = prompt_text.replace("<QueryAudios>", "included in concatenated stream")
        else:
            ref_parts = []
            for i, _ in enumerate(reference_audios):
                ref_parts.append(f"reference_audio_{i}: {speech_token}")
            ref_placeholder = ", ".join(ref_parts)

            query_parts = []
            for i, _ in enumerate(query_audios):
                query_parts.append(f"query_audio_{i}: {speech_token}")
            query_placeholder = ", ".join(query_parts)

            prompt_text = prompt_text.replace("<ReferenceAudios>", ref_placeholder)
            prompt_text = prompt_text.replace("<QueryAudios>", query_placeholder)
    
    if model_name in ["qwen_audio", "custom_qwen"]:
        return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt_text


def pick_prompt_template(
    prompt_templates: List[str],
    sample_index: int = 0,
    *,
    deterministic: bool = False,
) -> str:
    """Pick a prompt template; deterministic mode always uses the first template."""
    if not prompt_templates:
        raise ValueError("prompt_templates is empty")
    if deterministic:
        return prompt_templates[0]
    return random.choice(prompt_templates)


def collate_fn(batch: List[Dict[str, Any]], processor: Any = None, model_name: str = None, 
               prompt_templates: Optional[List[str]] = None, silence_delay_seconds: float = 0.5,
               deterministic_prompts: bool = False) -> Dict[str, Any]:
    
    audio_ids = [b["audio_id"] for b in batch]
    texts = [b["text"] for b in batch]
    
    # Check if this is SASV format (has reference_audios and query_audios)
    is_sasv_format = "reference_audios" in batch[0] and "query_audios" in batch[0]
    
    if is_sasv_format:
            # SASV format: multiple audios per sample
            reference_audios_list = []
            query_audios_list = []
            
            for b in batch:
                ref_audios = b.get("reference_audios", [])
                query_audios = b.get("query_audios", [])
                reference_audios_list.append(ref_audios)
                query_audios_list.append(query_audios)
            
            if (model_name in ("qwen_audio", "custom_qwen")) and processor:
                # Build prompts from templates - prompts are required for these models
                if not prompt_templates:
                    raise ValueError(
                        f"prompt_templates is required for {model_name}. "
                        "Configure Prompts.reasoning_prompts or Prompts.hard_label_prompts in config.yaml"
                    )
                
                prompts = []
                all_audios_per_sample = []
                
                for i, b in enumerate(batch):
                    template = pick_prompt_template(
                        prompt_templates, i, deterministic=deterministic_prompts
                    )
                    prompt = _format_prompt_for_model(
                        template, 
                        model_name, 
                        reference_audios=reference_audios_list[i],
                        query_audios=query_audios_list[i]
                    )
                    prompts.append(prompt)
                    
                    sample_audios = []
                    for wav_tensor in reference_audios_list[i] + query_audios_list[i]:
                        wav_np = wav_tensor.squeeze().numpy()
                        sample_audios.append(wav_np)
                    
                    if len(sample_audios) == 0:
                        sample_audios = [np.zeros(16000, dtype=np.float32)]
                    
                    all_audios_per_sample.append(sample_audios)
                
                if model_name in ("qwen_audio", "custom_qwen"):
                    batch_input_features = []
                    batch_input_ids = []
                    batch_attention_mask = []
                    batch_feature_attention_mask = []
                    
                    for prompt, sample_audios in zip(prompts, all_audios_per_sample):
                        sample_inputs = processor(
                            text=prompt,
                            audios=sample_audios,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True
                        )
                        batch_input_features.append(sample_inputs.get("input_features"))
                        batch_input_ids.append(sample_inputs.get("input_ids"))
                        batch_attention_mask.append(sample_inputs.get("attention_mask"))
                        batch_feature_attention_mask.append(sample_inputs.get("feature_attention_mask"))
                    
                    if batch_input_ids[0] is not None:
                        max_input_len = max(ids.shape[1] for ids in batch_input_ids)
                        pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
                        
                        padded_input_ids = []
                        padded_attention_mask = []
                        for ids, mask in zip(batch_input_ids, batch_attention_mask):
                            pad_len = max_input_len - ids.shape[1]
                            if pad_len > 0:
                                ids = torch.cat([ids, torch.full((ids.shape[0], pad_len), pad_token_id, dtype=ids.dtype)], dim=1)
                                mask = torch.cat([mask, torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype)], dim=1)
                            padded_input_ids.append(ids)
                            padded_attention_mask.append(mask)
                        batch_input_ids = padded_input_ids
                        batch_attention_mask = padded_attention_mask
                    
                    if batch_feature_attention_mask[0] is not None:
                        max_feature_len = max(mask.shape[1] if mask is not None else 0 for mask in batch_feature_attention_mask)
                        padded_feature_masks = []
                        for mask in batch_feature_attention_mask:
                            if mask is not None and mask.shape[1] < max_feature_len:
                                pad_len = max_feature_len - mask.shape[1]
                                mask = torch.cat([mask, torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype)], dim=1)
                            padded_feature_masks.append(mask)
                        batch_feature_attention_mask = padded_feature_masks
                    
                    return {
                        "audio_ids": audio_ids,
                        "input_features": torch.cat(batch_input_features, dim=0) if batch_input_features[0] is not None else None,
                        "input_ids": torch.cat(batch_input_ids, dim=0) if batch_input_ids[0] is not None else None,
                        "attention_mask": torch.cat(batch_attention_mask, dim=0) if batch_attention_mask[0] is not None else None,
                        "feature_attention_mask": torch.cat(batch_feature_attention_mask, dim=0) if batch_feature_attention_mask[0] is not None else None,
                        "text": texts,
                        "gt": [b.get("gt", "") for b in batch],
                        "answer": [b.get("answer", "") for b in batch],
                        "task": [b.get("task", "sasv") for b in batch],
                        "reasoning": [b.get("reasoning", "") for b in batch],
                        "reference_audios": reference_audios_list,
                        "query_audios": query_audios_list,
                        "raw_wav": [audios[0] if len(audios) > 0 else np.zeros(16000, dtype=np.float32) for audios in all_audios_per_sample],
                        "prompts": prompts
                    }
            elif model_name in ("sasv_w2v_aasist", "sasv_ecapa_w2v_aasist"):
                enroll_tensors: List[torch.Tensor] = []
                query_tensors: List[torch.Tensor] = []
                for i in range(len(batch)):
                    ref_list = reference_audios_list[i]
                    qry_list = query_audios_list[i]
                    ref_wav = ref_list[0].squeeze() if ref_list else torch.zeros(16000)
                    qry_wav = qry_list[0].squeeze() if qry_list else torch.zeros(16000)
                    enroll_tensors.append(ref_wav.float().flatten())
                    query_tensors.append(qry_wav.float().flatten())
                enroll_wav, enroll_padding_mask = _pad_raw_wav_batch(enroll_tensors)
                query_wav, query_padding_mask = _pad_raw_wav_batch(query_tensors)
                return {
                    "audio_ids": audio_ids,
                    "enroll_wav": enroll_wav,
                    "query_wav": query_wav,
                    "enroll_padding_mask": enroll_padding_mask,
                    "query_padding_mask": query_padding_mask,
                    "gt": [b.get("gt", "") for b in batch],
                    "answer": [b.get("answer", "") for b in batch],
                    "task": [b.get("task", "sasv") for b in batch],
                    "text": texts,
                    "reasoning": [b.get("reasoning", "") for b in batch],
                }
            else:
                # For SALMON/Whisper: concatenate with configurable silence delay
                sample_rate = 16000
                silence_samples = int(silence_delay_seconds * sample_rate)
                prompts = []
                if prompt_templates:
                    for i, _ in enumerate(batch):
                        template = pick_prompt_template(
                            prompt_templates, i, deterministic=deterministic_prompts
                        )
                        prompt = _format_prompt_for_model(
                            template,
                            model_name,
                            reference_audios=reference_audios_list[i],
                            query_audios=query_audios_list[i]
                        )
                        prompts.append(prompt)
                all_audios = []
                enroll_num_samples_list = []
                query_num_samples_list = []
                for i, b in enumerate(batch):
                    ref_audios = reference_audios_list[i]
                    query_audios = query_audios_list[i]
                    
                    # Track enroll/query sample counts for ArcFace extraction
                    enroll_samples = sum(w.shape[-1] for w in ref_audios)
                    query_samples = sum(w.shape[-1] for w in query_audios)
                    enroll_num_samples_list.append(enroll_samples)
                    query_num_samples_list.append(query_samples)
                    
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    combined_wav = None
                    for wav_tensor in ref_audios + query_audios:
                        wav_np = wav_tensor.squeeze().numpy()
                        if combined_wav is None:
                            combined_wav = wav_np
                        else:
                            combined_wav = np.concatenate([combined_wav, silence, wav_np])
                    
                    if combined_wav is None:
                        combined_wav = np.zeros(16000, dtype=np.float32)
                    
                    all_audios.append(combined_wav)
                
                if processor:
                    inputs = processor(all_audios, sampling_rate=16000, return_tensors="pt")
                    spectrograms = inputs.input_features
                else:
                    spectrograms = torch.randn(len(batch), 80, 3000)

                raw_wav, padding_mask = _pad_raw_wav_batch(all_audios)
                
                # Extract speaker_ids for both enroll and query
                enroll_speaker_ids = []
                query_speaker_ids = []
                for b in batch:
                    ref_spk = b.get("reference_speaker_ids", [])
                    q_spk = b.get("query_speaker_ids", [])
                    enroll_speaker_ids.append(ref_spk[0] if ref_spk else "")
                    query_speaker_ids.append(q_spk[0] if q_spk else "")

                return {
                    "audio_ids": audio_ids,
                    "spectrogram": spectrograms,
                    "raw_wav": raw_wav,
                    "padding_mask": padding_mask,
                    "gt": [b.get("gt", "") for b in batch],
                    "answer": [b.get("answer", "") for b in batch],
                    "task": [b.get("task", "sasv") for b in batch],
                    "text": texts,
                    "reasoning": [b.get("reasoning", "") for b in batch],
                    "enroll_speaker_ids": enroll_speaker_ids,
                    "query_speaker_ids": query_speaker_ids,
                    "enroll_num_samples": enroll_num_samples_list,
                    "query_num_samples": query_num_samples_list,
                    "silence_samples": silence_samples,
                    "prompts": prompts,
                }
    else:
        # Original format: single audio per sample
        raw_wavs = [b["raw_wav"].squeeze().numpy() for b in batch]
        
        if (model_name in ("qwen_audio", "custom_qwen")) and processor:
            if not prompt_templates:
                raise ValueError(
                    f"prompt_templates is required for {model_name}. "
                    "Configure Prompts.reasoning_prompts or Prompts.hard_label_prompts in config.yaml"
                )
            
            prompts = []
            for i, b in enumerate(batch):
                template = pick_prompt_template(
                    prompt_templates, i, deterministic=deterministic_prompts
                )
                prompt = _format_prompt_for_model(template, model_name)
                prompts.append(prompt)
            
            try:
                inputs = processor(text=prompts, audios=raw_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            except (TypeError, ValueError):
                try:
                    inputs = processor(text=prompts, audio=raw_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
                except (TypeError, ValueError):
                    inputs = processor(prompts, raw_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            
            return {
                "audio_ids": audio_ids,
                "input_features": inputs.get("input_features"),
                "input_ids": inputs.get("input_ids"), 
                "attention_mask": inputs.get("attention_mask"),
                "feature_attention_mask": inputs.get("feature_attention_mask"),
                "text": texts, 
                "is_bonafide": torch.tensor([b.get("is_bonafide", False) for b in batch]) if "is_bonafide" in batch[0] else None,
                "task": [b.get("task", "antispoofing") for b in batch],
                "reasoning": [b.get("reasoning", "") for b in batch],
                "raw_wav": raw_wavs,
                "prompts": prompts
            }

    if processor:
        inputs = processor(raw_wavs, sampling_rate=16000, return_tensors="pt")
        spectrograms = inputs.input_features
    else:
        spectrograms = torch.randn(len(batch), 80, 3000)
    
    return {
        "audio_ids": audio_ids,
        "spectrogram": spectrograms,
        "raw_wav": raw_wavs, 
        "is_bonafide": torch.tensor([b["is_bonafide"] for b in batch]),
        "task": [b["task"] for b in batch],
        "text": texts,
        "reasoning": [b["reasoning"] for b in batch]
    }
