# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM multimodal checkpoint loading and video sampling extensions."""

import json
import math

import numpy as np
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.glm5_next.image_processing_glm5_next import (
    Glm5NextImageProcessor,
    smart_resize,
)
from transformers.models.glm5_next.processing_glm5_next import (
    Glm5NextProcessor as HFGlm5NextProcessor,
)
from transformers.models.glm5_next.processing_glm5_next import (
    Glm5NextProcessorKwargs,
)
from transformers.models.glm5_next.video_processing_glm5_next import (
    Glm5NextVideoProcessor as HFGlm5NextVideoProcessor,
)
from transformers.processing_utils import MultiModalData
from transformers.utils import cached_file
from transformers.video_utils import VideoMetadata

_MAX_VIDEO_TOKENS = 30000
GLM_VIDEO_DEFAULT_FPS = 2.0
GLM_VIDEO_DEFAULT_MAX_FRAMES = 2048


def glm_sample_frame_indices(
    total_frames: int,
    fps: float,
    duration: float,
    *,
    target_fps: float | None = None,
    max_frame_count: int | None = None,
    temporal_patch_size: int = 2,
) -> list[int]:
    """GLM video frame sampling (training-reference parity).

    ``target_fps`` is the ``fps_interval`` request knob. The greedy walk
    advances at ``1 / (temporal_patch_size * target_fps)`` seconds, so on
    frame-dense sources it collects more candidates than ``extract_t`` and
    the ``> extract_t`` fixup re-spreads the picks uniformly with
    ``np.linspace`` -- that fallback is the intended reference behavior, not
    an accident. Short clips (fewer frames than ``extract_t``) are spread at
    evenly spaced timestamps (``floor`` sampling; the linspace variant
    samples frames unevenly and cost 4 points on video grounding evals).
    Request overrides: ``target_fps`` -> fps interval, ``max_frame_count``
    -> frame cap.
    """
    max_frame_idx = total_frames - 1
    if not duration:
        duration = (round(max_frame_idx / fps) + 1) if fps else 0
    if max_frame_count is None:
        max_frame_count = GLM_VIDEO_DEFAULT_MAX_FRAMES
    if target_fps is None:
        target_fps = GLM_VIDEO_DEFAULT_FPS

    extract_t = int(duration * target_fps)
    extract_t = min(extract_t, int(max_frame_count))

    duration_per_frame = 1 / fps
    timestamps = [i * duration_per_frame for i in range(total_frames)]
    max_second = int(duration)

    if total_frames < extract_t:
        frame_indices = [
            math.floor(_i * total_frames / extract_t) for _i in range(extract_t)
        ]
    else:
        frame_indices = []
        current_second = 0.0
        inv_fps = 1 / (temporal_patch_size * target_fps)
        for frame_index in range(total_frames):
            if timestamps[frame_index] >= current_second:
                current_second += inv_fps
                frame_indices.append(frame_index)
                if current_second >= max_second:
                    break

    if len(frame_indices) < extract_t:
        if len(frame_indices) == 0:
            start, end = 0, max(total_frames - 1, 0)
        else:
            start, end = frame_indices[0], frame_indices[-1]
        frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(frame_indices) > extract_t:
        frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

    seen, uniq = set(), []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            uniq.append(int(idx))

    if len(uniq) & 1:
        uniq.append(uniq[-1])

    return uniq


class Glm5NextVideoProcessor(HFGlm5NextVideoProcessor):
    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample frame indices with GLM's fps-interval policy.

        ``fps`` / ``target_fps``, ``max_frames`` and ``fps_interval`` /
        ``max_frame_count_dynamic`` are the overrides described in
        :func:`glm_sample_frame_indices`.
        """
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was "
                "provided which is required when sampling in GLM-5.3-Flash. Please "
                "pass in `VideoMetadata` object or set `do_sample_frames=False`."
            )

        target_fps = fps if fps is not None else kwargs.get("target_fps")
        if target_fps is None:
            target_fps = getattr(self, "fps_interval", self.fps)
        indices = glm_sample_frame_indices(
            metadata.total_num_frames,
            metadata.fps,
            metadata.duration or 0,
            target_fps=target_fps,
            max_frame_count=kwargs.get("max_frames")
            or getattr(self, "max_frame_count_dynamic", self.max_frames),
            temporal_patch_size=self.temporal_patch_size,
        )
        return np.array(indices)


class Glm5NextProcessor(HFGlm5NextProcessor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Build the processor directly from the checkpoint config.

        GLM-5.3-Flash stores nested image/video configs in
        ``processor_config.json`` and declares a custom processor class. This
        method reads those configs directly and caps only the video token budget.
        """
        from transformers import AutoTokenizer

        model_path = pretrained_model_name_or_path
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        config_kwargs = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "revision",
                "subfolder",
                "token",
            )
            if key in kwargs
        }

        def _cap_cfg(cfg: dict, *, is_video: bool) -> dict:
            # Video keeps a serving token cap (the checkpoint's 240k-token
            # budget would starve the KV cache at startup profiling); images
            # follow the checkpoint budget verbatim so preprocessing matches
            # the HF reference exactly.
            if is_video and cfg.get("max_image_tokens") is not None:
                cfg["max_image_tokens"] = min(
                    cfg["max_image_tokens"], _MAX_VIDEO_TOKENS
                )
            return cfg

        ip_cfg = _cap_cfg(
            dict(get_image_processor_config(model_path, **config_kwargs)),
            is_video=False,
        )
        image_processor = Glm5NextImageProcessor(
            **{k: v for k, v in ip_cfg.items() if k != "image_processor_type"}
        )

        processor_config_file = cached_file(
            model_path,
            "processor_config.json",
            **config_kwargs,
        )
        if processor_config_file is None:
            raise FileNotFoundError(
                f"processor_config.json was not found for {model_path!r}"
            )
        with open(processor_config_file) as f:
            vp_cfg = _cap_cfg(dict(json.load(f)["video_processor"]), is_video=True)
        video_processor = Glm5NextVideoProcessor(
            **{k: v for k, v in vp_cfg.items() if k != "video_processor_type"}
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = dict(
                Glm5NextProcessorKwargs._defaults.get("images_kwargs", {})
            )
            images_kwargs.update(kwargs)
            merge_size = (
                images_kwargs.get("merge_size") or self.image_processor.merge_size
            )

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(
                    *image_size, images_kwargs
                )
                for image_size in image_sizes
            ]
            num_image_tokens = [(n // merge_size**2) for n in num_image_patches]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        if video_sizes is not None:
            videos_kwargs = dict(
                Glm5NextProcessorKwargs._defaults.get("videos_kwargs", {})
            )
            videos_kwargs.update(kwargs)
            merge_size = (
                videos_kwargs.get("merge_size") or self.video_processor.merge_size
            )
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(
                    *video_size, videos_kwargs
                )
                for video_size in video_sizes
            ]
            num_video_tokens = [(n // merge_size**2) for n in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)


__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextVideoProcessor",
    "Glm5NextProcessor",
    "smart_resize",
]
