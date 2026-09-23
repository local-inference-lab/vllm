# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading

from vllm.entrypoints.serve.utils.task_dump import format_asyncio_tasks


def test_dump_follows_a_request_through_async_generators():
    """The dump reaches the awaited event through nested async generators and
    shows local names and types, but not the values of text locals."""

    async def main() -> str:
        released = asyncio.Event()

        async def engine_outputs(request_id: str):
            prompt = "private prompt text"
            num_tokens = 228652
            await released.wait()
            yield prompt, num_tokens

        async def stream_chunks(request_id: str):
            async for output in engine_outputs(request_id):
                yield output

        async def respond():
            async for _ in stream_chunks("chatcmpl-held"):
                pass

        task = asyncio.create_task(respond(), name="held-response")
        await asyncio.sleep(0)
        dump = format_asyncio_tasks()
        released.set()
        await task
        return dump

    stop = threading.Event()
    helper = threading.Thread(target=stop.wait, name="tokenizer-worker")
    helper.start()
    try:
        dump = asyncio.run(main())
    finally:
        stop.set()
        helper.join()

    section = dump.split("Task 'held-response':")[1].split("  Task ")[0]
    order = [
        section.index(name)
        for name in ("respond", "stream_chunks", "engine_outputs", "Event.wait")
    ]
    assert order == sorted(order)
    assert "request_id='chatcmpl-held'" in section
    assert "num_tokens=228652" in section
    assert "prompt: str" in section
    assert "private prompt text" not in dump
    assert "task is blocked on <Future pending" in section
    assert "Thread 'tokenizer-worker'" in dump
