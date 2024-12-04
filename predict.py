import logging
import os
import sys

sys.path.append("/HunyuanVideo/")

from pathlib import Path
from typing import List, Optional
import torch
from cog import BasePredictor, Input, Path as CogPath
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid


_log = logging.getLogger("cog_predictor")


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("This model requires CUDA")

        args = parse_args()
        _log.info(args)
        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")

        # Create save folder to save the samples
        save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
        Path(save_path).mkdir(save_path, exist_ok=True, parents=True)

        self.args = args

        # Load models
        self.unyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt for video generation",
            default="A cat walks on the grass, realistic style."
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default=None
        ),
        width: int = Input(
            description="Video width",
            default=1280,
            ge=1,
        ),
        height: int = Input(
            description="Video height",
            default=720,
            ge=1,
        ),
        video_length: int = Input(
            description="Number of frames",
            default=129,
            ge=1,
            le=129
        ),
        seed: int = Input(
            description="Random seed",
            default=None
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale",
            default=6.0,
            ge=1.0,
            le=6.0,
        ),
        embedded_guidance_scale: float = Input(
            description="Embedded guidance scale",
            default=6.0,
            ge=1.0,
            le=6.0,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=1,
        ),
        flow_shift: float = Input(
            description="Flow shift parameter",
            default=7.0,
        ),
        flow_reverse: bool = Input(
            description="Whether to reverse flow",
            default=True
        ),
    ) -> List[CogPath]:
        """Run video generation inference"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        _log.warning(f"Seed is not set. Using seed: {seed}")

        self.hunyuan_video_sampler.args.video_size = [height, width]

        # Generate videos
        outputs = self.model.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=video_length,
            seed=seed,
            negative_prompt=negative_prompt,
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=1,
            batch_size=1,
            flow_shift=flow_shift,
            flow_reverse=flow_reverse,
            embedded_guidance_scale=embedded_guidance_scale
        )
        
        # Save videos and return paths
        output_paths = []
        output_dir = Path(self.args.save_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        sample = outputs["samples"][0]

        # Save the generated video
        sample = sample.unsqueeze(0)
        output_path = output_dir / "video.mp4"
        save_videos_grid(sample, output_path, fps=24)

        return CogPath(output_path)
