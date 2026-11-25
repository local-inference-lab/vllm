# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
import subprocess
import sys
from array import array
from pathlib import Path

import msgspec
import numpy as np
import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import _with_packed_prompt
from vllm.v1.request import Request, RequestStatus, TokenBuffer
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_copies_session_id_from_engine_core_request():
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session-1",
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.session_id == "session-1"


def test_rust_frontend_prompt_ids_survive_engine_admission():
    # Use the same request payloads as the Rust client's wire compatibility test.
    fixtures = (
        Path(__file__).resolve().parents[2]
        / "rust/src/engine-core-client/src/tests/python_compat.py"
    )
    if not fixtures.parent.is_dir():
        pytest.skip("Rust client tests are absent from this checkout")
    payloads = subprocess.check_output([sys.executable, str(fixtures)], text=True)
    decoder = MsgpackDecoder(EngineCoreRequest)
    expected = [("req-1", [11, 22]), ("req-defaults", [5, 6, 7])]
    for payload, (request_id, ids) in zip(
        payloads.splitlines()[:2], expected, strict=True
    ):
        received = decoder.decode([bytes.fromhex(payload)])
        request = Request.from_engine_core_request(received, block_hasher=None)
        assert request.request_id == request_id
        assert list(request.prompt_token_ids) == ids
        assert list(request.all_token_ids) == ids


@pytest.mark.parametrize("ids", [None, [], [0], [2**31 - 1, 70000, 1, 0]])
def test_list_prompt_ids_round_trip(ids: list[int] | None):
    wire = msgspec.msgpack.encode(["id", ids, None, {}, None, 0.0, None, None, None])
    received = MsgpackDecoder(EngineCoreRequest).decode([wire])
    if ids is None:
        assert received.prompt_token_ids is None
    else:
        assert list(received.prompt_token_ids) == ids


@pytest.mark.parametrize(
    "invalid", [[True], [1, 2.5], [1, "2"], ["1", 2, 3], [2**31], [-(2**31) - 1]]
)
def test_list_prompt_ids_reject_invalid_values(invalid):
    wire = msgspec.msgpack.encode(
        ["id", invalid, None, {}, None, 0.0, None, None, None]
    )
    with pytest.raises(msgspec.ValidationError):
        MsgpackDecoder(EngineCoreRequest).decode([wire])


@pytest.mark.parametrize(
    "descriptor",
    [
        ["x", [], b""],
        ["<i4", [1], "data"],
        ["<i4", [1], 0],
        ["<i4", [1], -1],
        ["<i4", [1], 2],
        ["<i4", [True], 1],
        ["<i4", [1], True],
        ["<i4", [-1], 1],
    ],
)
def test_invalid_packed_descriptor_fails_prompt_validation(descriptor):
    wire = msgspec.msgpack.encode(
        ["id", descriptor, None, {}, None, 0.0, None, None, None]
    )
    with pytest.raises(msgspec.ValidationError, match="Expected `int`"):
        MsgpackDecoder(EngineCoreRequest).decode([wire, bytes(4)])


@pytest.mark.parametrize("dtype", ["<i4", ">i4", "<u4", ">u4"])
@pytest.mark.parametrize("ids", [[], [0, 70000, 2**31 - 1]])
def test_packed_prompt_accepts_32_bit_integer_encodings(dtype, ids):
    descriptor = [dtype, [len(ids)], np.asarray(ids, dtype=dtype).tobytes()]
    wire = msgspec.msgpack.encode(
        ["id", descriptor, None, {}, None, 0.0, None, None, None]
    )
    received = MsgpackDecoder(EngineCoreRequest).decode([wire])
    request = Request.from_engine_core_request(received, block_hasher=None)
    assert request.prompt_token_ids == ids


@pytest.mark.parametrize("dtype", ["<u4", ">u4"])
@pytest.mark.parametrize("threshold", [1, 1 << 20], ids=["out-of-band", "inline"])
def test_packed_unsigned_prompt_overflow_fails_during_decode(dtype, threshold):
    ids = np.array([0, 2**31, 1], dtype=dtype)
    frames = MsgpackEncoder(size_threshold=threshold).encode(
        ["id", ids, None, {}, None, 0.0, None, None, None]
    )
    with pytest.raises(msgspec.ValidationError, match="signed int32"):
        MsgpackDecoder(EngineCoreRequest).decode(frames)


@pytest.mark.parametrize("threshold, frame_count", [(1, 2), (1 << 20, 1)])
def test_packed_prompt_owns_tokens_across_engine_admission(threshold, frame_count):
    tokens = [1001, 2002] * 4096
    frontend = EngineCoreRequest(
        request_id="packed",
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    outbound = _with_packed_prompt(frontend)
    tokens[0] = 3003
    encoded = MsgpackEncoder(size_threshold=threshold).encode(outbound)
    assert len(encoded) == frame_count
    received = MsgpackDecoder(EngineCoreRequest).decode(encoded)
    assert isinstance(received.prompt_token_ids, np.ndarray)
    assert received.prompt_token_ids.dtype == np.int32
    if frame_count == 2:
        assert np.shares_memory(
            received.prompt_token_ids, np.frombuffer(encoded[1], dtype=np.int32)
        )
    request = Request.from_engine_core_request(received, block_hasher=None)
    assert frontend.prompt_token_ids is tokens
    assert request.prompt_token_ids[0] == 1001

    frames: list[pickle.PickleBuffer] = []
    snapshot = pickle.dumps(
        request._all_token_ids, protocol=5, buffer_callback=frames.append
    )
    request.append_output_token_ids(4004)
    assert list(pickle.loads(snapshot, buffers=frames)) == [1001, 2002] * 4096
    assert request.all_token_ids[-1] == 4004
    assert len(request.prompt_token_ids) == 8192


def test_token_buffer_sequence_operations():
    assert list(TokenBuffer()) == []
    tokens = TokenBuffer(iter([1, 70000, 129279]))
    assert tokens[-1] == 129279
    with pytest.raises(IndexError):
        _ = tokens[3]
    sliced = tokens[::-2]
    copied = tokens.copy()
    assert type(sliced) is list and sliced == [129279, 1]
    assert type(copied) is list and copied == [1, 70000, 129279]

    assert tokens == (1, 70000, 129279)
    assert tokens != (1, 129279, 70000)
    assert tokens + [7] == [1, 70000, 129279, 7]
    assert [7] + tokens == [7, 1, 70000, 129279]

    alias = tokens
    tokens += [3, 4]
    tokens[0] = 11
    assert tokens is alias
    assert tokens == [11, 70000, 129279, 3, 4]
    assert copied == [1, 70000, 129279]
    assert sliced == [129279, 1]
    assert isinstance(tokens, array) and tokens.itemsize == 4


def test_token_buffer_compares_ndarray_as_one_sequence():
    tokens = TokenBuffer([1, 2])
    assert (tokens == np.array([1, 2])) is True
    assert (tokens != np.array([1, 3])) is True
    assert (tokens == np.array([[1, 2]])) is False
    assert (tokens == iter([1, 2])) is False


@pytest.mark.parametrize(
    "dtype", [np.dtype("=i4"), np.dtype("=i4").newbyteorder("S"), np.dtype("=u4")]
)
def test_token_buffer_owns_strided_array_values(dtype: np.dtype):
    source = np.array([0, 99, 70000, 99, 2**31 - 1], dtype=dtype)
    tokens = TokenBuffer(source[::2])
    assert tokens == [0, 70000, 2**31 - 1]
    source[2] = 17
    tokens[0] = 42
    assert tokens[1] == 70000
    assert source[0] == 0


def test_token_buffer_rejects_out_of_range_ids():
    with pytest.raises(OverflowError):
        TokenBuffer([2**31])
    with pytest.raises(OverflowError):
        TokenBuffer(np.array([2**31], dtype=np.uint32))


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.float32),
    ],
    ids=["matrix", "int64", "float32"],
)
def test_token_buffer_rejects_invalid_arrays(invalid: np.ndarray):
    with pytest.raises(ValueError, match="1D array of 32-bit integer token IDs"):
        TokenBuffer(invalid)


@pytest.mark.parametrize("protocol", [4, 5])
def test_token_buffer_pickle_round_trip(protocol: int):
    original = TokenBuffer([1, 70000, 129279])
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is TokenBuffer
    assert restored == original
    restored[0] = 42
    assert original[0] == 1


def test_token_buffer_pickle_owns_out_of_band_snapshot():
    tokens = TokenBuffer([1, 70000])
    frames: list[pickle.PickleBuffer] = []
    payload = pickle.dumps(tokens, protocol=5, buffer_callback=frames.append)
    (frame,) = frames
    with frame.raw() as view:
        receiver_frame = bytearray(view)
        tokens[0] = 42
        tokens.append(7)
        assert bytes(view) == receiver_frame
    restored = pickle.loads(payload, buffers=[receiver_frame])
    receiver_frame[:] = bytes(len(receiver_frame))
    assert type(restored) is TokenBuffer
    assert restored == [1, 70000]
