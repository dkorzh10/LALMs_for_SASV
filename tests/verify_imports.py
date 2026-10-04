import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def test_core_imports():
    from models.sasv_salmon import SASVSalmonModel  # noqa: F401
    from models.qwen_audio import QwenAudioModel  # noqa: F401
    from trainers.sft_trainer import SFTTrainer  # noqa: F401
    from trainers.grpo_trainer import GRPOTrainer  # noqa: F401


if __name__ == "__main__":
    test_core_imports()
