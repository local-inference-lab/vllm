# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""격리된 Kimi worker 소스에 선택적 메모리 감사 훅을 설치한다.

연구용으로 준비된 소스 패치를 보존한다. K3PATCH_GPU_WORKER로 지정한 파일의
정확한 init_device 앵커만 수정하며, 현재 프로세스나 GPU에는 접근하지 않는다.
설치된 worker가 K3_MEMORY_AUDIT=1로 시작할 때만 allocator history와 SIGUSR1
처리를 활성화한다. 출력 디렉터리는 K3_MEMORY_AUDIT_DIR로 지정한다.

이 훅은 CUDA의 비공개 snapshot API를 사용하고 신호 처리 중 GPU API를 호출한다.
메모리 사용량과 타이밍에 영향을 주므로 유휴 상태의 격리된 진단 서버에서만
사용한다. SIGUSR1 재진입과 실제 CUDA 실행은 별도 검증 대상이다. pickle에는
스택 경로가 들어갈 수 있으며, 신뢰하지 않는 snapshot은 역직렬화하지 않는다.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_GPU_WORKER",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu_worker.py",
    )
)
MARKER = "kimi-k3-memory-audit"

OLD = """    def init_device(self):
        if self.device_config.device_type == "cuda":
"""

NEW = '''    def _k3_memory_audit_install(self):
        """kimi-k3-memory-audit: allocator 이력과 SIGUSR1 snapshot을 기록한다."""
        import os as _os
        import pickle as _pickle
        import signal as _signal
        import time as _time

        import torch as _torch

        if _os.environ.get("K3_MEMORY_AUDIT", "0") != "1":
            return
        out_dir = _os.environ.get("K3_MEMORY_AUDIT_DIR", "/cache/memory-audit")
        try:
            _torch.cuda.memory._record_memory_history(
                max_entries=int(
                    _os.environ.get("K3_MEMORY_AUDIT_MAX_ENTRIES", "400000")
                )
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("메모리 감사 이력 기록을 사용할 수 없습니다: %s", exc)

        def _dump(signum, frame):
            try:
                _os.makedirs(out_dir, exist_ok=True)
                rank = getattr(self, "rank", _os.environ.get("RANK", "x"))
                stamp = _time.strftime("%Y%m%d-%H%M%S")
                free, total = _torch.cuda.mem_get_info()
                payload = {
                    "rank": rank,
                    "time": stamp,
                    "mem_get_info": {"free": int(free), "total": int(total)},
                    "memory_stats": dict(_torch.cuda.memory_stats()),
                    "snapshot": _torch.cuda.memory._snapshot(),
                }
                path = _os.path.join(out_dir, f"rank{rank}-{stamp}.pickle")
                with open(path, "wb") as fh:
                    _pickle.dump(payload, fh)
                logger.info("메모리 감사 snapshot 기록: %s (여유 %.2f / 전체 %.2f GiB)",
                            path, free / 2**30, total / 2**30)
            except Exception as exc:  # pragma: no cover
                logger.warning("메모리 감사 snapshot 기록 실패: %s", exc)

        _signal.signal(_signal.SIGUSR1, _dump)
        logger.info("메모리 감사 활성화: SIGUSR1 -> %s", out_dir)

    def init_device(self):
        self._k3_memory_audit_install()
        if self.device_config.device_type == "cuda":
'''


def main() -> int:
    """대상 소스의 앵커를 검증하고 한 번만 감사 훅을 삽입한다.

    Returns:
        패치 완료 또는 이미 패치된 경우 0.

    Raises:
        SystemExit: 대상 앵커가 정확히 하나가 아닌 경우.
        SyntaxError: 패치된 소스가 Python 문법에 맞지 않는 경우.
        OSError: 대상 파일을 읽거나 쓸 수 없는 경우.
    """
    text = PATH.read_text()
    if MARKER in text:
        print(f"이미 설치된 worker 메모리 감사: {MARKER}")
        return 0
    if text.count(OLD) != 1:
        raise SystemExit(f"대상 앵커가 정확히 하나가 아닙니다: {PATH}")
    patched = text.replace(OLD, NEW, 1)
    compile(patched, str(PATH), "exec")
    PATH.write_text(patched)
    print(f"worker 메모리 감사 설치 완료: {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
