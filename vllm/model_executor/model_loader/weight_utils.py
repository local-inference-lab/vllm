# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for downloading and initializing model weights."""

import asyncio
import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import filelock
import huggingface_hub.constants
import numpy as np
import regex as re
import torch
from safetensors.torch import load, load_file, safe_open, save_file
from tqdm.auto import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm import envs
from vllm.config import ModelConfig
from vllm.config.load import (
    DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
    DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    LoadConfig,
)
from vllm.distributed import get_tensor_model_parallel_rank, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from vllm.model_executor.model_loader.ep_weight_filter import (
    should_skip_weight,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import hf_api, hf_fs
from vllm.utils.import_utils import PlaceholderModule

try:
    from runai_model_streamer import SafetensorsStreamer
except ImportError:
    runai_model_streamer = PlaceholderModule("runai_model_streamer")  # type: ignore[assignment]
    SafetensorsStreamer = runai_model_streamer.placeholder_attr("SafetensorsStreamer")

try:
    from fastsafetensors import SingleGroup
except ImportError:
    fastsafetensors = PlaceholderModule("fastsafetensors")
    SingleGroup = fastsafetensors.placeholder_attr("SingleGroup")

from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least

logger = init_logger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


def enable_xet_high_performance():
    """automatically activates xet high performance mode"""
    if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
        huggingface_hub.constants.HF_XET_HIGH_PERFORMANCE = True


enable_xet_high_performance()


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


@contextmanager
def atomic_writer(
    filepath: str | Path, mode: str = "w", encoding: str | None = None
) -> Generator[IO]:
    """
    Context manager that provides an atomic file writing routine.

    The context manager writes to a temporary file and, if successful,
    atomically replaces the original file.

    Args:
        filepath (str or Path): The path to the file to write.
        mode (str): The file mode for the temporary file (e.g., 'w', 'wb').
        encoding (str): The encoding for text mode.

    Yields:
        file object: A handle to the temporary file.
    """
    # Create a temporary file in the same directory as the target file
    # to ensure it's on the same filesystem for an atomic replace.
    temp_dir = os.path.dirname(filepath)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

    try:
        # Open the temporary file for writing
        with os.fdopen(temp_fd, mode=mode, encoding=encoding) as temp_file:
            yield temp_file

        # If the 'with' block completes successfully,
        # perform the atomic replace.
        os.replace(temp_path, filepath)

    except Exception:
        logger.exception(
            "Error during atomic write. Original file '%s' not modified", filepath
        )
        raise
    finally:
        # Clean up the temporary file if it still exists.
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _natural_sort_key(filepath: str) -> list:
    """Natural sort key for filenames with numeric components, such as
    model-00001-of-00005.safetensors -> ['model-', 1, '-of-', 5, '.safetensors']"""
    return [
        int(s) if s.isdigit() else s
        for s in re.split(r"(\d+)", os.path.basename(filepath))
    ]


def maybe_download_from_modelscope(
    model: str,
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    allow_patterns: list[str] | str | None = None,
) -> str | None:
    """Download model from ModelScope hub if VLLM_USE_MODELSCOPE is True.

    Returns the path to the downloaded model, or None if the model is not
    downloaded from ModelScope."""
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model, download_dir):
            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=ignore_patterns,
                    allow_patterns=allow_patterns,
                )
            else:
                model_path = model
        return model_path
    return None


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


# TODO(woosuk): Move this to other place.
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    if model_config.quantization is None:
        raise ValueError("Model quantization method is not specified in the config.")
    quant_cls = get_quantization_config(model_config.quantization)

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)

    # Pipe information about heads to enable TP-aware loading of attn_head scales
    if (
        hf_quant_config is not None
        and hf_quant_config.get("quant_method") == "compressed-tensors"
        and "config_groups" in hf_quant_config
    ):
        n_heads = model_config.model_arch_config.total_num_attention_heads
        n_kv_heads = model_config.model_arch_config.total_num_kv_heads

        hf_quant_config["total_num_heads"] = n_heads
        hf_quant_config["total_num_kv_heads"] = (
            n_kv_heads if n_kv_heads is not None else n_heads
        )

    if hf_quant_config is not None:
        # `model_config.quantization_config` may be set alongside a checkpoint
        # quant config: the checkpoint determines `quant_cls`, and the user's
        # QuantizationConfigArgs is consulted by individual quant methods
        # (e.g. for activation overrides via the MXFP4 oracle).

        # For modelopt_mixed, config.json's quantization_config may or may
        # not contain the per-layer quantized_layers map.  Newer checkpoints
        # embed it directly; older ones keep it only in hf_quant_config.json.
        # If it is missing, fall through to the file-based loading path.
        if (
            model_config.quantization == "modelopt_mixed"
            and "quantized_layers" not in hf_quant_config
        ):
            pass  # fall through to file-based loading below
        else:
            return quant_cls.from_config(hf_quant_config)

    # if hf_quant_config is None, we will try to get config from
    # hf_overrides
    hf_overrides = model_config.hf_overrides
    if not isinstance(hf_overrides, dict):
        raise ValueError(
            "hf_overrides must be a dict for get_quant_config "
            "to get the quantization config from it."
        )
    quantization_config_file = hf_overrides.get("quantization_config_file", None)
    if quantization_config_file is not None:
        if hasattr(quant_cls, "from_config_file"):
            return quant_cls.from_config_file(quantization_config_file)
        else:
            raise NotImplementedError(
                "from_config_file is specified in hf_override config, "
                "but quant_cls.from_config_file is not implemented in "
                f"{quant_cls}"
            )
    quantization_config_json = hf_overrides.get("quantization_config_dict_json", None)
    if quantization_config_json is not None:
        if hasattr(quant_cls, "from_config_dict_json"):
            return quant_cls.from_config_dict_json(quantization_config_json)
        else:
            raise NotImplementedError(
                "from_config_dict_json is specified in hf_override config, "
                "but quant_cls.from_config_dict_json is not implemented in "
                f"{quant_cls}"
            )

    # Online quantization doesn't read from checkpoint configs - it quantizes
    # fp16/bf16 weights on the fly during loading.
    if model_config.quantization_config is not None:
        from vllm.config.quantization import QuantizationConfigArgs
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        assert isinstance(model_config.quantization_config, QuantizationConfigArgs)
        return OnlineQuantizationConfig(args=model_config.quantization_config)

    model_name_or_path = (
        maybe_download_from_modelscope(
            model_config.model,
            revision=model_config.revision,
            download_dir=load_config.download_dir,
            allow_patterns=["*.json"],
        )
        or model_config.model
    )
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_config.model, load_config.download_dir):
            hf_folder = hf_api().snapshot_download(
                model_config.model,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)

        if model_config.quantization in ("modelopt", "modelopt_mixed"):
            if config.get("producer", {}).get("name") == "modelopt":
                return quant_cls.from_config(config)
            else:
                raise ValueError(
                    f"Unsupported quantization config"
                    f" found for {model_config.quantization} in {f}."
                )

    return quant_cls.from_config(config)


def get_sparse_attention_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> dict[str, Any]:
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = hf_api().snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    config_file = os.path.join(hf_folder, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    logger.info("Loaded sparse attention config from %s", config_file)

    return config


@instrument(span_name="Download weights - HF")
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    subfolder: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        subfolder (Optional[str]): The subfolder within the model repository
            to download weights from.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Attempt to reduce allow_patterns to a single pattern
        # so we only have to call snapshot_download once.
        try:
            fs = hf_fs()
            file_list = fs.ls(
                os.path.join(model_name_or_path, subfolder or ""),
                detail=False,
                revision=revision,
            )

            # If downloading safetensors and an index file exists, use the
            # specific file names from the index to avoid downloading
            # unnecessary files (e.g., from subdirectories like "original/").
            index_file = f"{model_name_or_path}/{SAFE_WEIGHTS_INDEX_NAME}"
            if "*.safetensors" in allow_patterns and index_file in file_list:
                index_path = hf_api().hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=SAFE_WEIGHTS_INDEX_NAME,
                    cache_dir=cache_dir,
                    revision=revision,
                    subfolder=subfolder,
                )
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
                if weight_map:
                    # Extra [] so that weight_map files are treated as a
                    # single allow_pattern in the loop below
                    allow_patterns = [list(set(weight_map.values()))]  # type: ignore[list-item]
                else:
                    allow_patterns = ["*.safetensors"]
            else:
                # Use the first pattern found in the HF repo's files.
                for pattern in allow_patterns:
                    if fnmatch.filter(file_list, pattern):
                        allow_patterns = [pattern]
                        break
        except Exception as e:
            logger.warning(
                "Failed to get file list for '%s'. Trying each pattern in "
                "allow_patterns individually until weights have been "
                "downloaded. Error: %s",
                model_name_or_path,
                e,
            )

    logger.debug("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        for allow_pattern in allow_patterns:
            hf_folder = hf_api().snapshot_download(
                model_name_or_path,
                allow_patterns=allow_pattern,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=local_only,
            )
            # If we have downloaded weights for this allow_pattern,
            # we don't need to check the rest.
            # allow_pattern can be a list (from weight_map) or str (glob)
            if isinstance(allow_pattern, list):
                break
            if any(Path(hf_folder).glob(allow_pattern)):
                break
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder


def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: str | None,
    subfolder: str | None = None,
    revision: str | None = None,
) -> None:
    """Download hf safetensors index file from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        index_file (str): The safetensors index file name
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        subfolder (Optional[str]): The subfolder within the model repository
            to download weights from.
        revision (Optional[str]): The revision of the model.
    """
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        try:
            # Download the safetensors index file.
            hf_api().hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                subfolder=subfolder,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        # If file not found on remote or locally, we should not fail since
        # only some models will have index_file.
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.info("No %s found in local cache.", index_file)
        except huggingface_hub.utils.EntryNotFoundError:
            logger.info("No %s found in remote.", index_file)


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Check if files referenced in model.safetensors.index.json actually exist.
    # Raise error if any file is missing.
    hf_weights_files_set = set(hf_weights_files)
    missing_files = weight_files_in_index - hf_weights_files_set
    if missing_files:
        raise FileNotFoundError(
            f"Weight files referenced in index but missing: {missing_files}"
        )
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def enable_tqdm(use_tqdm_on_load: bool):
    return use_tqdm_on_load and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: str | None,
    hf_folder: str,
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.

    Will dump the model weights to numpy files if they are not already dumped.
    """
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: list[str] = []
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file) as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)


def _get_checkpoints_size_bytes(files: list[str]) -> int:
    """Return the total size of the checkpoint files in bytes."""
    if not files:
        return 0
    return sum(os.path.getsize(f) for f in files)


def _get_available_ram_bytes() -> int:
    """Return available RAM, honoring cgroup limits."""
    import psutil

    host_available = psutil.virtual_memory().available

    from vllm.utils.cpu_resource_utils import get_cgroup_memory_limit

    cgroup_limit, cgroup_usage = get_cgroup_memory_limit()
    if cgroup_limit is None:
        return host_available
    cgroup_available = (
        cgroup_limit if cgroup_usage is None else max(0, cgroup_limit - cgroup_usage)
    )
    return min(host_available, cgroup_available)


def _get_fs_type(files: list[str]) -> str:
    """Get the filesystem type of the first file in *files* (Linux only)."""
    if not files:
        return ""
    try:
        # Only the first file is checked — all checkpoint shards reside
        # in the same directory and therefore on the same filesystem.
        resolved = os.path.realpath(files[0])
        best_mount = ""
        best_fstype = ""
        # /proc/mounts may contain nested mount points (e.g. "/" -> ext4,
        # "/data" -> nfs4, "/data/local" -> ext4).  We pick the entry with
        # the longest matching mount_point — the same "longest prefix match"
        # rule the kernel uses to decide which filesystem serves a path.
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fstype = parts[1], parts[2]
                if (
                    resolved == mount_point
                    or resolved.startswith(os.path.join(mount_point, ""))
                ) and len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = fstype
        return best_fstype
    except Exception:
        # /proc/mounts is Linux-specific; on other OSes (or if the read
        # fails for any reason) we fall back to an empty string.
        return ""


def _prefetch_checkpoint(
    file_path: str,
    block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> None:
    """Prefetch a checkpoint file into the OS page cache.

    Reads the file in blocks so the kernel caches its pages before workers load
    the same file.
    """
    if block_size < 1:
        raise ValueError("safetensors prefetch block size must be >= 1")

    with open(file_path, "rb") as f:
        while f.read(block_size):
            pass


def _prefetch_all_checkpoints(
    sorted_files: list[str],
    num_prefetch_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> None:
    """Start prefetching checkpoint files into page cache in a background thread."""
    if num_prefetch_threads < 1:
        raise ValueError("safetensors prefetch num threads must be >= 1")
    if block_size < 1:
        raise ValueError("safetensors prefetch block size must be >= 1")

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    paths_to_prefetch = sorted_files[rank::world_size]
    total_for_rank = len(paths_to_prefetch)

    async def _prefetch_all() -> None:
        loop = asyncio.get_running_loop()
        completed = 0
        next_log_pct = 10

        async def prefetch_one(
            path: str,
            executor: concurrent.futures.ThreadPoolExecutor,
        ) -> None:
            nonlocal completed, next_log_pct
            try:
                await loop.run_in_executor(
                    executor, _prefetch_checkpoint, path, block_size
                )
                completed += 1
                if total_for_rank > 0 and next_log_pct <= 100:
                    pct = 100 * completed / total_for_rank
                    if pct >= next_log_pct:
                        logger.info(
                            "Prefetching checkpoint files: %d%% (%d/%d)",
                            next_log_pct,
                            completed,
                            total_for_rank,
                        )
                        next_log_pct += 10
            except Exception:
                logger.warning(
                    "Failed to prefetch checkpoint file %r.", path, exc_info=True
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_prefetch_threads
        ) as executor:
            await asyncio.gather(
                *(prefetch_one(p, executor) for p in paths_to_prefetch)
            )

    def _run_prefetch() -> None:
        start = time.perf_counter()
        asyncio.run(_prefetch_all())
        elapsed = time.perf_counter() - start
        logger.info(
            "Prefetching checkpoint files into page cache finished in %.2fs",
            elapsed,
        )

    logger.info(
        "Prefetching checkpoint files into page cache started "
        "(in background, num_threads=%d, block_size=%d bytes)",
        num_prefetch_threads,
        block_size,
    )
    threading.Thread(target=_run_prefetch, daemon=True).start()


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str | None = None,
    local_expert_ids: set[int] | None = None,
    *,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    safetensors_prefetch_block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files.

    When *local_expert_ids* is provided, expert weights not belonging to
    this rank are skipped **before** reading from disk, which drastically
    reduces storage I/O for MoE models under EP.
    """
    loading_desc = "Loading safetensors checkpoint shards"
    if safetensors_load_strategy == "eager":
        loading_desc += " (eager)"

    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)

    fs_type = _get_fs_type(sorted_files)
    is_net_fs = fs_type in ("nfs", "nfs4", "lustre")
    total_bytes = _get_checkpoints_size_bytes(sorted_files)
    avail_bytes = _get_available_ram_bytes()
    ram_threshold_pct = 90
    fits_in_ram = total_bytes <= (ram_threshold_pct / 100.0) * avail_bytes
    fs_name = fs_type.upper() if fs_type else "unknown"

    logger.info_once(
        "Filesystem type for checkpoints: %s. Checkpoint size: %.2f GiB. "
        "Available RAM: %.2f GiB.",
        fs_name,
        total_bytes / 1024**3,
        avail_bytes / 1024**3,
    )

    should_prefetch = safetensors_load_strategy == "prefetch"
    if safetensors_load_strategy is None:
        if is_net_fs and fits_in_ram:
            should_prefetch = True
        elif is_net_fs and not fits_in_ram:
            logger.warning_once(
                "Network filesystem (%s) detected but checkpoint total size "
                "(%.2f GiB) exceeds %d%% of available RAM (%.2f GiB). "
                "Skipping auto-prefetch.",
                fs_name,
                total_bytes / 1024**3,
                ram_threshold_pct,
                avail_bytes / 1024**3,
            )
        elif not is_net_fs and fits_in_ram:
            logger.info_once(
                "Auto-prefetch is disabled because the filesystem (%s) is not a "
                "recognized network FS (NFS/Lustre). If you want to force "
                "prefetching, start vLLM with --safetensors-load-strategy=prefetch.",
                fs_name,
            )
        elif not is_net_fs and not fits_in_ram:
            logger.info_once(
                "Auto-prefetch is disabled because the filesystem (%s) is not a "
                "recognized network FS (NFS/Lustre) and the checkpoint size "
                "(%.2f GiB) exceeds %d%% of available RAM (%.2f GiB).",
                fs_name,
                total_bytes / 1024**3,
                ram_threshold_pct,
                avail_bytes / 1024**3,
            )
    elif should_prefetch and not fits_in_ram:
        logger.warning_once(
            "safetensors_load_strategy='prefetch' was explicitly specified, but "
            "checkpoint total size (%.2f GiB) exceeds %d%% of available RAM "
            "(%.2f GiB). This may cause out-of-memory errors.",
            total_bytes / 1024**3,
            ram_threshold_pct,
            avail_bytes / 1024**3,
        )

    if should_prefetch:
        _prefetch_all_checkpoints(
            sorted_files,
            num_prefetch_threads=safetensors_prefetch_num_threads,
            block_size=safetensors_prefetch_block_size,
        )

    leftover_state_dict: dict[str, torch.Tensor] = {}
    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
        elif safetensors_load_strategy == "torchao":
            # we can't load flattened torchao tensor subclasses directly into the model
            # instead we reconstruct the subclasses here before returning
            if not torchao_version_at_least("0.15.0"):
                raise ValueError(
                    "Please use torchao version >= 0.15.0 "
                    "to load torchao safetensors checkpoint"
                )
            from torchao.prototype.safetensors.safetensors_support import (
                unflatten_tensor_state_dict,
            )

            with safe_open(st_file, framework="pt") as f:
                state_dict = {}
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    state_dict[name] = f.get_tensor(name)

                # update with leftover tensor data from previous iteration, if any
                state_dict.update(leftover_state_dict)
                metadata = f.metadata()
                # due to sharded checkpoints, we are not guaranteed that we have all
                # tensor subclass data on one file
                # state_dict has the leftover data from this step and we wait for
                # missing information to be provided in a future iteration
                unflattened_state_dict, leftover_state_dict = (
                    unflatten_tensor_state_dict(state_dict, metadata)
                )
            yield from unflattened_state_dict.items()
        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param


def multi_thread_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model safetensor files."""

    def _load_file(st_file: str):
        result = load_file(st_file, device="cpu")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Note to use generator here so we do not store all the loaded files in memory
        # at the same time, which can cause OOM for large models.
        futures = (executor.submit(_load_file, st_file) for st_file in hf_weights_files)
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        for future in futures_iter:
            state_dict = future.result()
            del future
            for key in list(state_dict):
                yield key, state_dict.pop(key)


def runai_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    is_distributed: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    with SafetensorsStreamer() as streamer:
        is_cuda_alike = current_platform.is_cuda_alike()
        device = (
            f"cuda:{current_platform.current_device()}"
            if is_distributed and is_cuda_alike
            else "cpu"
        )

        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
        total_tensors = sum(
            len(tensors_meta)
            for tensors_meta in streamer.files_to_tensors_metadata.values()
        )

        tensor_iter = tqdm(
            streamer.get_tensors(),
            total=total_tensors,
            desc="Loading safetensors using Runai Model Streamer",
            bar_format=_BAR_FORMAT,
            disable=not enable_tqdm(use_tqdm_on_load),
            mininterval=2,
        )

        for name, tensor in tensor_iter:
            yield name, tensor.clone()


def fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files
    using fastsafetensor library.

    Uses ParallelLoader for pipelined loading: the producer thread
    prepares metadata for the next shard while the consumer yields
    tensors from the current shard.
    """
    from fastsafetensors.parallel_loader import ParallelLoader

    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    device = torch.device(f"cuda:{current_platform.current_device()}")
    hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)

    # Use nogds=True for TP > 1 to avoid cuFileDriverOpen() which
    # initializes the GDS DMA subsystem for all visible GPUs, creating
    # unwanted CUDA contexts on every device.
    nogds = pg.size() > 1

    queue_size = envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE
    tqdm_enabled = enable_tqdm(use_tqdm_on_load)

    def _make_loader(nogds: bool) -> "ParallelLoader":
        return ParallelLoader(
            pg=pg,
            hf_weights_files=hf_weights_files,
            queue_size=queue_size,
            use_tqdm_on_load=tqdm_enabled,
            device=str(device),
            nogds=nogds,
        )

    # GDS can fail either at construction or lazily inside the producer
    # thread during iteration (e.g. cuFileHandleRegister returning
    # CU_FILE_HANDLE_NOT_REGISTERED on a filesystem without GDS support).
    # Catch both and fall back to nogds, but only before yielding any
    # tensor -- restarting mid-stream would reload earlier shards.
    pl = None
    yielded = False
    try:
        try:
            pl = _make_loader(nogds)
            for name, tensor in pl.iterate_weights():
                yielded = True
                yield name, tensor
        except RuntimeError as e:
            if nogds or yielded or "gds" not in str(e):
                raise
            logger.warning_once(
                "GDS not enabled, setting `nogds=True`.\n"
                "For more information, see: https://github.com/foundation-model-stack/"
                "fastsafetensors?tab=readme-ov-file#basic-api-usages"
            )
            if pl is not None:
                pl.close()
            pl = _make_loader(nogds=True)
            yield from pl.iterate_weights()
    finally:
        if pl is not None:
            pl.close()


def _matches_weight_name_prefixes(
    weight_name: str, weight_name_prefixes: Sequence[str]
) -> bool:
    """Return whether a checkpoint tensor matches a selected name prefix."""
    return weight_name.startswith(tuple(weight_name_prefixes))


@dataclass(frozen=True)
class _InstantTensorCpuFallback:
    position: int
    name: str
    filename: str


@dataclass(frozen=True)
class _InstantTensorSelection:
    gpu_tensor_count: int
    selected_tensor_count: int
    cpu_fallbacks: tuple[_InstantTensorCpuFallback, ...]


def instanttensor_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    weight_name_prefixes: Sequence[str] | None = None,
    *,
    indexed_tensor_files: dict[str, str] | None = None,
    priority_weight_name_prefixes: Sequence[str] | None = None,
    small_checkpoint_max_bytes: int | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream selected safetensors through bounded InstantTensor staging.

    Indexed tensors that exceed ``INSTANTTENSOR_BUFFER_SIZE`` are read through
    CPU safetensors at their original checkpoint positions. Priority prefixes
    are emitted through CPU safetensors before GPU staging begins. Setting
    ``INSTANTTENSOR_COPY=0`` exposes borrowed staging views; deferred consumers
    must clone such tensors before advancing the iterator.
    """
    try:
        import instanttensor
    except ImportError as e:
        raise ImportError(
            "Please install instanttensor via `pip install vllm[instanttensor]`"
        ) from e

    if not current_platform.is_cuda():
        raise ValueError("InstantTensor requires NVIDIA GPUs")

    if small_checkpoint_max_bytes is not None and (
        isinstance(small_checkpoint_max_bytes, bool) or small_checkpoint_max_bytes <= 0
    ):
        raise ValueError(
            "InstantTensor small-checkpoint threshold must be a positive integer"
        )

    if priority_weight_name_prefixes:
        if not all(
            isinstance(prefix, str) and prefix
            for prefix in priority_weight_name_prefixes
        ):
            raise ValueError(
                "InstantTensor priority weight prefixes must be non-empty strings"
            )
        priority_names_by_file: dict[str, set[str]] = defaultdict(set)
        tensor_files: Iterable[tuple[str, str]]
        if indexed_tensor_files is not None:
            tensor_files = indexed_tensor_files.items()
        else:
            # The target and every speculative-model source share one loader
            # configuration. Header discovery lets an unindexed draft ignore
            # target-only priorities without disabling them for the target.
            discovered_tensor_files: list[tuple[str, str]] = []
            for filename in hf_weights_files:
                absolute_filename = os.path.abspath(filename)
                with safe_open(filename, framework="pt", device="cpu") as reader:
                    discovered_tensor_files.extend(
                        (name, absolute_filename) for name in reader.offset_keys()
                    )
            tensor_files = discovered_tensor_files

        for name, filename in tensor_files:
            if not name.startswith(tuple(priority_weight_name_prefixes)):
                continue
            if weight_name_prefixes and not _matches_weight_name_prefixes(
                name, weight_name_prefixes
            ):
                continue
            priority_names_by_file[os.path.abspath(filename)].add(name)
        if not priority_names_by_file:
            logger.info_once(
                "InstantTensor priority prefixes %s match no tensors in this "
                "checkpoint source; all selected tensors use the normal "
                "loading schedule.",
                tuple(priority_weight_name_prefixes),
            )
        else:
            logger.info_once(
                "InstantTensor loads %d tensors from %d/%d checkpoint shards "
                "through CPU safetensors before GPU staging because they match "
                "priority prefixes %s.",
                sum(len(names) for names in priority_names_by_file.values()),
                len(priority_names_by_file),
                len(hf_weights_files),
                tuple(priority_weight_name_prefixes),
            )
        priority_names: set[str] = set()
        for filename in hf_weights_files:
            names = priority_names_by_file.get(os.path.abspath(filename))
            if not names:
                continue
            with safe_open(filename, framework="pt", device="cpu") as reader:
                physical_names = list(reader.offset_keys())
                missing_names = names.difference(physical_names)
                if missing_names:
                    raise RuntimeError(
                        "Safetensors index maps priority tensors to a shard "
                        "that does not contain them: "
                        f"{sorted(missing_names)[:3]} in {filename}"
                    )
                for name in physical_names:
                    if name not in names:
                        continue
                    priority_names.add(name)
                    yield name, reader.get_tensor(name)
        expected_priority_names = set().union(*priority_names_by_file.values())
        if priority_names != expected_priority_names:
            missing_names = expected_priority_names.difference(priority_names)
            raise RuntimeError(
                "InstantTensor priority tensors reference checkpoint shards "
                f"outside the selected file set: {sorted(missing_names)[:3]}"
            )
    else:
        priority_names = set()

    if (
        small_checkpoint_max_bytes is not None
        and indexed_tensor_files is None
        and not priority_names
    ):
        checkpoint_size = _get_checkpoints_size_bytes(hf_weights_files)
        if checkpoint_size <= small_checkpoint_max_bytes:
            logger.info_once(
                "Loading the %.2f GiB unindexed checkpoint through CPU "
                "safetensors because it does not exceed the configured %.2f GiB "
                "InstantTensor small-checkpoint threshold.",
                checkpoint_size / 1024**3,
                small_checkpoint_max_bytes / 1024**3,
            )
            yield from safetensors_weights_iterator(
                hf_weights_files,
                use_tqdm_on_load,
            )
            return

    try:
        world_group = get_world_group()
    except AssertionError:
        # Entering here only in unit tests where the world group is not initialized.
        process_group = None
    else:
        process_group = world_group.device_group if world_group.world_size > 1 else None

    device = current_platform.current_device()
    copy_setting = os.getenv("INSTANTTENSOR_COPY", "1")
    if copy_setting not in {"0", "1"}:
        raise ValueError(f"INSTANTTENSOR_COPY must be 0 or 1, got {copy_setting!r}")
    copy_tensors = copy_setting == "1"

    configured_buffer_size = os.getenv("INSTANTTENSOR_BUFFER_SIZE")
    max_gpu_tensor_size: int | None = None
    if configured_buffer_size is not None:
        try:
            max_gpu_tensor_size = int(configured_buffer_size)
        except ValueError as e:
            raise ValueError(
                "INSTANTTENSOR_BUFFER_SIZE must be an integer number of bytes"
            ) from e
        if max_gpu_tensor_size <= 0:
            raise ValueError("INSTANTTENSOR_BUFFER_SIZE must be greater than zero")

    restrict_before_io = (
        indexed_tensor_files is not None
        or bool(weight_name_prefixes)
        or max_gpu_tensor_size is not None
        or bool(priority_names)
    )
    open_kwargs: dict[str, Any] = {}
    if restrict_before_io:
        open_kwargs["load_now"] = False

    # copy=True is the safe default. copy=False exposes views into
    # InstantTensor's reusable buffer for synchronous consumers.
    instant_open = instanttensor.safe_open(
        list(hf_weights_files),
        framework="pt",
        device=device,
        process_group=process_group,
        copy=copy_tensors,
        **open_kwargs,
    )
    selection: _InstantTensorSelection | None = None
    if restrict_before_io:
        selection = _restrict_instanttensor_to_selected_ranges(
            instant_open,
            indexed_tensor_files=indexed_tensor_files,
            weight_name_prefixes=weight_name_prefixes,
            max_tensor_size=max_gpu_tensor_size,
            excluded_tensor_names=priority_names,
        )
        if selection.selected_tensor_count == 0:
            return

    cpu_fallbacks = selection.cpu_fallbacks if selection is not None else ()
    if not cpu_fallbacks:
        with instant_open as f:
            # Track bytes so the bar reports load throughput (GB/s).
            pbar = tqdm(
                total=f.total_tensor_size,
                desc="Loading safetensors using InstantTensor loader",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
                position=tqdm._get_free_pos(),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                mininterval=1.0,
            )
            try:
                tensor: torch.Tensor | None = None
                for name, tensor in f.tensors():
                    pbar.update(tensor.numel() * tensor.element_size())
                    if weight_name_prefixes and not _matches_weight_name_prefixes(
                        name, weight_name_prefixes
                    ):
                        continue
                    if not copy_tensors:
                        # Parameter loaders consume borrowed tensors synchronously.
                        # A deferred loader must own the tensor before advancing.
                        tensor._vllm_instanttensor_borrowed = True
                    yield name, tensor
                # A generator frame retains its final DLPack-backed view. Drop it
                # before the InstantTensor context releases the staging ring.
                tensor = None
            finally:
                pbar.close()
        return

    assert max_gpu_tensor_size is not None
    assert selection is not None
    logger.info_once(
        "Loading %d tensors larger than the %d-byte InstantTensor buffer "
        "through CPU safetensors",
        len(cpu_fallbacks),
        max_gpu_tensor_size,
    )
    fallback_by_position = {fallback.position: fallback for fallback in cpu_fallbacks}
    with ExitStack() as stack:
        fallback_readers: dict[str, Any] = {}
        for cpu_fallback in cpu_fallbacks:
            if cpu_fallback.filename not in fallback_readers:
                fallback_readers[cpu_fallback.filename] = stack.enter_context(
                    safe_open(cpu_fallback.filename, framework="pt", device="cpu")
                )

        pbar = None
        gpu_tensors: Any = iter(())
        if selection.gpu_tensor_count:
            instant_reader = stack.enter_context(instant_open)
            pbar = tqdm(
                total=instant_reader.total_tensor_size,
                desc="Loading safetensors using InstantTensor loader",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
                position=tqdm._get_free_pos(),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                mininterval=1.0,
            )
            gpu_tensors = iter(instant_reader.tensors())

        try:
            gpu_tensor: torch.Tensor | None = None
            for position in range(selection.selected_tensor_count):
                scheduled_fallback = fallback_by_position.get(position)
                if scheduled_fallback is not None:
                    yield (
                        scheduled_fallback.name,
                        fallback_readers[scheduled_fallback.filename].get_tensor(
                            scheduled_fallback.name
                        ),
                    )
                    continue
                try:
                    name, gpu_tensor = next(gpu_tensors)
                except StopIteration as e:
                    raise RuntimeError(
                        "InstantTensor produced fewer tensors than the selected "
                        "checkpoint schedule"
                    ) from e
                if pbar is not None:
                    pbar.update(gpu_tensor.numel() * gpu_tensor.element_size())
                if not copy_tensors:
                    gpu_tensor._vllm_instanttensor_borrowed = True
                yield name, gpu_tensor

            if next(gpu_tensors, None) is not None:
                raise RuntimeError(
                    "InstantTensor produced more tensors than the selected "
                    "checkpoint schedule"
                )
            gpu_tensor = None
        finally:
            if pbar is not None:
                pbar.close()


def _restrict_instanttensor_to_selected_ranges(
    instant_open: Any,
    *,
    indexed_tensor_files: dict[str, str] | None,
    weight_name_prefixes: Sequence[str] | None,
    max_tensor_size: int | None = None,
    excluded_tensor_names: set[str] | None = None,
) -> _InstantTensorSelection:
    """Restrict an InstantTensor loader to requested checkpoint ranges.

    Args:
        instant_open: Unopened InstantTensor safetensors loader whose metadata
            can be restricted before I/O starts.
        indexed_tensor_files: Mapping from tensor names to their canonical
            checkpoint files. ``None`` accepts tensors from every input file.
        weight_name_prefixes: Model parameter prefixes selected for loading.
            ``None`` or an empty sequence accepts every prefix.
        max_tensor_size: Largest tensor payload, in bytes, retained in the GPU
            staging path. Larger selected tensors are assigned to CPU loading.
        excluded_tensor_names: Tensor names already emitted by an earlier
            loading pass. Excluded names are removed before InstantTensor opens
            its GPU staging context.

    Returns:
        Selected GPU tensor count, total selected tensor count, and CPU
        fallbacks with their positions in checkpoint order.
    """
    required_attrs = (
        "filename",
        "ordered_tensor_metadatas",
        "tensor_offsets",
        "tensor_sizes",
        "total_tensor_size",
        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
    )
    missing = [name for name in required_attrs if not hasattr(instant_open, name)]
    if missing:
        raise RuntimeError(
            "Installed InstantTensor does not expose the metadata layout needed "
            f"for index-aware loading (missing: {', '.join(missing)})"
        )
    if instant_open.loader_handle is not None:
        raise RuntimeError("InstantTensor selection must be applied before opening I/O")

    filenames = list(instant_open.filename)
    metadata = list(instant_open.ordered_tensor_metadatas)
    offsets = list(instant_open.tensor_offsets)
    selected_filenames: list[str] = []
    selected_metadata: list[tuple[str, dict[str, Any]]] = []
    selected_offsets: list[tuple[int, int]] = []
    cpu_fallbacks: list[_InstantTensorCpuFallback] = []
    selected_position = 0
    excluded_selected_count = 0
    metadata_pos = 0
    offset_pos = 0

    for filename in filenames:
        filename_abs = os.path.abspath(filename)
        with safe_open(filename, framework="pt") as physical_file:
            physical_names = list(physical_file.offset_keys())
        tensor_count = len(physical_names)
        file_metadata = metadata[metadata_pos : metadata_pos + tensor_count]
        file_offsets = offsets[offset_pos : offset_pos + tensor_count + 1]
        if len(file_metadata) != tensor_count or len(file_offsets) != tensor_count + 1:
            raise RuntimeError(
                "InstantTensor metadata/offset count does not match safetensors "
                f"header for {filename}"
            )
        if [name for name, _ in file_metadata] != physical_names:
            raise RuntimeError(
                "InstantTensor tensor order does not match safetensors offset order "
                f"for {filename}"
            )

        keep = []
        for item_index, name in enumerate(physical_names):
            indexed_here = indexed_tensor_files is None or (
                indexed_tensor_files.get(name) == filename_abs
            )
            prefix_matches = not weight_name_prefixes or _matches_weight_name_prefixes(
                name, weight_name_prefixes
            )
            otherwise_selected = indexed_here and prefix_matches
            excluded_here = (
                otherwise_selected
                and excluded_tensor_names is not None
                and name in excluded_tensor_names
            )
            excluded_selected_count += int(excluded_here)
            selected_here = otherwise_selected and not excluded_here
            item_data_offsets = file_metadata[item_index][1]["data_offsets"]
            tensor_size = int(item_data_offsets[1]) - int(item_data_offsets[0])
            use_cpu_fallback = (
                selected_here
                and max_tensor_size is not None
                and tensor_size > max_tensor_size
            )
            keep.append(selected_here and not use_cpu_fallback)
            if selected_here:
                if use_cpu_fallback:
                    cpu_fallbacks.append(
                        _InstantTensorCpuFallback(
                            position=selected_position,
                            name=name,
                            filename=filename,
                        )
                    )
                selected_position += 1

        run_start = 0
        while run_start < tensor_count:
            while run_start < tensor_count and not keep[run_start]:
                run_start += 1
            if run_start == tensor_count:
                break
            run_end = run_start + 1
            while run_end < tensor_count and keep[run_end]:
                run_end += 1

            logical_file_index = len(selected_filenames)
            selected_filenames.append(filename)
            selected_metadata.extend(file_metadata[run_start:run_end])
            selected_offsets.extend(
                (logical_file_index, int(file_offsets[index][1]))
                for index in range(run_start, run_end)
            )
            selected_offsets.append((logical_file_index, int(file_offsets[run_end][1])))
            run_start = run_end

        metadata_pos += tensor_count
        offset_pos += tensor_count + 1

    if metadata_pos != len(metadata) or offset_pos != len(offsets):
        raise RuntimeError(
            "InstantTensor layout contains unaccounted metadata or offsets"
        )
    if not selected_metadata and not cpu_fallbacks and excluded_selected_count == 0:
        raise RuntimeError("InstantTensor index/prefix selection matched no tensors")

    selected_names = [name for name, _ in selected_metadata]
    if len(selected_names) != len(set(selected_names)):
        raise RuntimeError(
            "InstantTensor index-aware selection still contains duplicate tensor names"
        )
    fallback_names = [fallback.name for fallback in cpu_fallbacks]
    if len(fallback_names) != len(set(fallback_names)):
        raise RuntimeError(
            "InstantTensor CPU fallback selection contains duplicate tensor names"
        )
    overlap = set(selected_names).intersection(fallback_names)
    if overlap:
        raise RuntimeError(
            f"InstantTensor GPU and CPU selections overlap: {sorted(overlap)[:3]}"
        )
    selected_sizes = [
        int(item["data_offsets"][1]) - int(item["data_offsets"][0])
        for _, item in selected_metadata
    ]
    if selected_metadata:
        instant_open.filename = selected_filenames
        instant_open.ordered_tensor_metadatas = selected_metadata
        instant_open.tensor_name_to_index = {
            name: index for index, name in enumerate(selected_names)
        }
        instant_open.tensor_offsets = selected_offsets
        instant_open.tensor_sizes = selected_sizes
        instant_open.total_tensor_size = sum(selected_sizes)
        instant_open._determine_buffer_size(None)

    return _InstantTensorSelection(
        gpu_tensor_count=len(selected_metadata),
        selected_tensor_count=selected_position,
        cpu_fallbacks=tuple(cpu_fallbacks),
    )


def pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )
        yield from state.items()
        del state


def multi_thread_pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model bin/pt files."""

    def _load_file(bin_file: str):
        return torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_load_file, bin_file) for bin_file in hf_weights_files
        ]
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading pt checkpoint shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        for future in futures_iter:
            state = future.result()
            yield from state.items()
            del state


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # reshape to match before copying
            param.data.copy_(loaded_weight.view(param.shape))
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """Load weights that are row-parallelized."""
    tp_rank = get_tensor_model_parallel_rank()
    shard_dim = 0 if param.dim() != 1 else None

    if shard_dim is not None:
        shard_size = param.data.shape[shard_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)

    return default_weight_loader(param, loaded_weight)


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    return loader


def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:
    """Create a weight loader that post-processes the weights after loading"""

    def composed_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        loader(param, loaded_weight)
        param.data.copy_(fn(param))
        return

    return composed_loader


def initialize_dummy_weights(
    model: torch.nn.Module,
    model_config: ModelConfig,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.
    """
    for param in model.state_dict().values():
        initialize_single_dummy_weight(param, low, high, seed)


@torch.no_grad()
def initialize_single_dummy_weight(
    param: torch.Tensor,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    if param.device.type == "meta":
        return  # deferred to finalize_layerwise_processing (e.g. online quant)

    if not torch.is_floating_point(param):
        if current_platform.is_rocm():
            # On ROCm, integer params (e.g. GPTQ qweight/qzeros) are left
            # as torch.empty() by default, giving non-deterministic values
            # across processes. Zero them for reproducibility.
            param.zero_()
        return

    if current_platform.is_tpu():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        # Note: The param.uniform_ function cannot be used in this
        # context because it demands more TPU HBM than directly copying
        # from a CPU tensor.
        # Note: We avoid using torch.rank_like as it doesn't currently
        # support the generator argument.
        param.copy_(
            (high - low)
            * torch.rand(
                param.shape,
                generator=generator,
                dtype=param.dtype,
                layout=param.layout,
                requires_grad=param.requires_grad,
                device="cpu",
            )
            + low
        )
        torch._sync(param)
        return

    generator = torch.Generator(device=param.data.device)
    generator.manual_seed(seed)
    if torch.finfo(param.data.dtype).bits < 16:
        # uniform_ doesn't support < 16-bit datatypes (FP8)
        dtype = param.data.dtype
        tmp_param = param.data.to(torch.float16)
        tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
        param.data.copy_(tmp_param)
    else:
        param.uniform_(low, high, generator=generator)


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Remap the name of FP8 k/v_scale parameters.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
    """
    # Already in vLLM's expected form (e.g. weights pre-renamed by a
    # `WeightsMapper` from the quant config). Skip the regex remap, which
    # would otherwise double-apply the `.attn` prefix and drop the weight.
    if name in params_dict:
        return name
    if name.endswith(".kv_scale"):
        logger.warning_once(
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"
        )
        # NOTE: we remap the deprecated kv_scale to k_scale
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            logger.warning_once(
                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                name,
                remapped_name,
            )
            return None
        return remapped_name

    if any("mla_attn" in key for key in params_dict):
        attn_str = "mla_attn.mla_attn"
        logger.debug_once(
            f"Found mla_attn with k_scale and v_scale in "
            f"the checkpoint, using {attn_str} as attn_str"
        )
    else:
        attn_str = "attn"
    # Define scale name mapping patterns in order of precedence
    scale_mapping_patterns = [
        # ModelOpt format: .self_attn.{k,v}_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (
            r"\.self_attn\.([kv])_proj\.([kv])_scale$",
            rf".self_attn.{attn_str}.\2_scale",
        ),
        # QKV proj format: .self_attn.qkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # Qwen3 MoE format: .self_attn.qkqkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkqkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # NemotronH format: .mixer.{k,v}_proj.{k,v}_scale ->
        # .mixer.attn.{k,v}_scale
        (r"\.mixer\.[kv]_proj\.([kv])_scale$", r".mixer.attn.\1_scale"),
        # HYV3 format: .self_attn.q.scale -> .self_attn.attn.q_scale
        (r"\.self_attn\.q\.scale$", r".self_attn.attn.q_scale"),
        # HYV3 format: .self_attn.{k,v}_cache.scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.([kv])_cache\.scale$", r".self_attn.attn.\1_scale"),
        # Default format: .{k,v}_scale -> .attn.{k,v}_scale
        (r"\.([qkv])_scale$", r".attn.\1_scale"),
        (r"\.([qkv])_zero_point$", r".attn.\1_zero_point"),
    ]

    # Check if name ends with k_scale or v_scale
    if name.endswith(
        (
            ".k_scale",
            ".v_scale",
            ".q_scale",
            ".k_zero_point",
            ".v_zero_point",
            ".q_zero_point",
            ".q.scale",
            ".k_cache.scale",
            ".v_cache.scale",
        )
    ):
        import regex as re

        for pattern, replacement in scale_mapping_patterns:
            if re.search(pattern, name):
                remapped_name = re.sub(pattern, replacement, name)
                if remapped_name not in params_dict:
                    scale_type = name.split(".")[-1]
                    logger.warning_once(
                        "Found %s in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). %s is not loaded.",  # noqa: E501
                        scale_type,
                        name,
                        remapped_name,
                        scale_type,
                    )
                    return None
                return remapped_name

    # If there were no matches, return the untouched param name
    return name


def maybe_remap_moe_expert_param_name(
    name: str,
    params_dict: dict[str, torch.nn.Parameter],
) -> str:
    """
    Remap MoE expert parameter names to account for routed_experts hierarchy.

    This handles the transition from the old FusedMoE structure where weights
    were directly in the experts module, to the new MoERunner → RoutedExperts
    structure.

    Checkpoint weights have names like:
        layers.0.mlp.experts.w13_weight
        layers.0.feed_forward.experts.w2_input_scale
    But actual parameters are now:
        layers.0.mlp.experts.routed_experts.w13_weight
        layers.0.feed_forward.experts.routed_experts.w2_input_scale

    This function inserts 'routed_experts.' into the path when needed.

    Args:
        name: Parameter name from checkpoint
        params_dict: Dictionary of model parameters (from named_parameters())

    Returns:
        Remapped parameter name if routed_experts hierarchy exists,
        otherwise the original name
    """
    # Only remap if this looks like an expert parameter
    if ".experts." not in name:
        return name

    # Skip if already has routed_experts
    if ".experts.routed_experts." in name:
        return name

    # Expert parameter patterns to check
    expert_param_suffixes = [
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_input_scale",
        "w2_input_scale",
        "w13_bias",
        "w2_bias",
        "w13_scale",
        "w2_scale",
        "w13_g_idx",
        "w2_g_idx",
        "w13_qweight",
        "w2_qweight",
        "w13_qzeros",
        "w2_qzeros",
        "w13_weight_shape",
        "w2_weight_shape",
    ]

    # Check if this is an expert weight parameter
    is_expert_param = any(
        f".{suffix}" in name or name.endswith(suffix)
        for suffix in expert_param_suffixes
    )

    if not is_expert_param:
        return name

    # Try inserting routed_experts after .experts.
    new_name = name.replace(".experts.", ".experts.routed_experts.", 1)

    # Only use the new name if it exists in the model
    if new_name in params_dict:
        return new_name

    # Otherwise return original name (old checkpoint format or different structure)
    return name


def remap_moe_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    params_dict: dict[str, torch.nn.Parameter],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Wrapper generator that remaps MoE expert parameter names for backward compatibility.

    This allows models with custom weight loading to automatically handle both old
    and new checkpoint formats without needing model-specific remapping code.

    Usage:
        params_dict = dict(model.named_parameters())
        for name, weight in remap_moe_expert_weights(weights, params_dict):
            # name is automatically remapped if needed
            param = params_dict[name]
            ...

    Args:
        weights: Iterator of (name, tensor) tuples from checkpoint
        params_dict: Dictionary of model parameters (from named_parameters())

    Yields:
        (remapped_name, tensor) tuples
    """
    for name, weight in weights:
        remapped_name = maybe_remap_moe_expert_param_name(name, params_dict)
        yield (remapped_name, weight)
