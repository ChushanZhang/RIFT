import hashlib
import os
from collections import OrderedDict
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from rift.utils.logging_config import get_logger
from rift.utils import misc
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        expected_total_episodes=None,
        exclude_episodes=None,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        sample_video_before_transforms: bool = False,
        text_memory_cache_size: int = 128,
        vae_cache_index_dir=None,
        vae_cache_dir=None,
    ):
        if type(text_memory_cache_size) is not int or text_memory_cache_size < 0:
            raise ValueError("text_memory_cache_size must be a nonnegative integer.")
        self.text_memory_cache_size = text_memory_cache_size
        self._text_memory_cache = OrderedDict()
        self._text_memory_cache_pid = os.getpid()
        self._latent_reader = None
        self.vae_cache_namespace = None
        if vae_cache_index_dir is not None:
            if vae_cache_dir is None:
                raise ValueError("Indexed latent loading requires vae_cache_dir.")
            if global_sample_stride != 1:
                raise ValueError("Indexed latent loading requires global_sample_stride=1.")
            if processor is None:
                raise ValueError("Indexed latent loading requires RIFTProcessor.")
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            expected_total_episodes=(
                OmegaConf.to_container(expected_total_episodes, resolve=True)
                if isinstance(expected_total_episodes, DictConfig)
                else expected_total_episodes
            ),
            exclude_episodes=(
                OmegaConf.to_container(exclude_episodes, resolve=True)
                if isinstance(exclude_episodes, DictConfig)
                else exclude_episodes
            ),
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))
        self.sample_video_before_transforms = sample_video_before_transforms

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(vae_cache_index_dir is None)
        self.lerobot_dataset.strict_loading = vae_cache_index_dir is not None

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if self.sample_video_before_transforms and processor is None:
            raise ValueError("Early video sampling requires an image processor.")
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if vae_cache_index_dir is not None:
                from .processors.rift_processor import RIFTProcessor
                if type(processor) is not RIFTProcessor:
                    raise TypeError("Indexed latent loading requires RIFTProcessor.")
                processor.validate_deterministic_image_transforms()
            if self.sample_video_before_transforms:
                from .processors.rift_processor import RIFTProcessor
                if type(processor) is not RIFTProcessor:
                    raise TypeError("Early video sampling requires RIFTProcessor.")
                processor.set_image_sample_indices(self.video_sample_indices)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        if vae_cache_index_dir is not None:
            from rift.utils.indexed_vae_cache import IndexedVaeLatentReader
            self._latent_reader = IndexedVaeLatentReader(
                index_dir=vae_cache_index_dir,
                cache_dir=vae_cache_dir,
                dataset_dirs=dataset_dirs,
                num_frames=num_frames,
                video_size=video_size,
                video_sample_indices=self.video_sample_indices,
                concat_multi_camera=concat_multi_camera,
                processor=processor,
            )
            self.vae_cache_namespace = self._latent_reader.namespace
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"][self.video_sample_indices]
        if getattr(self, "_latent_reader", None) is None:
            video = self._get_video(sample)
            visual_data = {"video": video}
            video_frames = video.shape[1]
        else:
            location = tuple(int(sample[key]) for key in
                             ("dataset_index", "episode_index", "frame_index"))
            visual_data = {
                "video_latents": self._latent_reader.load(*location),
                "video_shape": torch.tensor(
                    [3, len(self.video_sample_indices), *self.video_size], dtype=torch.long
                ),
                "vae_cache_namespace": self.vae_cache_namespace,
                "vae_cache_keys": self._latent_reader.key(*location),
            }
            video_frames = len(self.video_sample_indices)

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video_frames <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got {video_frames}")
        if action.shape[0] % (video_frames - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video_frames - 1}"
            )

        task = sample["instruction"]

        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            **visual_data,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        return data

    def _get_video(self, sample):
        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.sample_video_before_transforms:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.sample_video_before_transforms:
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        return video

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        if self._text_memory_cache_pid != os.getpid():
            self._text_memory_cache.clear()
            self._text_memory_cache_pid = os.getpid()
        while len(self._text_memory_cache) > self.text_memory_cache_size:
            self._text_memory_cache.popitem(last=False)
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        try:
            file_stat = os.stat(cache_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            ) from None
        cache_key = os.path.abspath(cache_path)
        signature = (file_stat.st_mtime_ns, file_stat.st_ctime_ns, file_stat.st_size,
                     file_stat.st_dev, file_stat.st_ino)
        cached = self._text_memory_cache.pop(cache_key, None)
        if self.text_memory_cache_size and cached is not None and cached[0] == signature:
            self._text_memory_cache[cache_key] = cached
            return cached[1].clone(), cached[2].clone()
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        if self.text_memory_cache_size:
            self._text_memory_cache[cache_key] = (signature, context, context_mask)
            while len(self._text_memory_cache) > self.text_memory_cache_size:
                self._text_memory_cache.popitem(last=False)
            # _get clears padded context positions in place; keep cached bytes intact.
            return context.clone(), context_mask.clone()
        return context, context_mask

    def __getitem__(self, idx):
        if getattr(self, "_latent_reader", None) is not None:
            return self._get(idx)
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
