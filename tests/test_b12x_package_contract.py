from pathlib import Path


def test_runtime_does_not_reference_legacy_sparkinfer_package() -> None:
    runtime_root = Path(__file__).parents[1] / "vllm"
    legacy_markers = ("sparkinfer.", "SPARKINFER_")
    # PR #228 owns the EXL3 runtime port because it replaces this file in full.
    deferred_to_paired_pr = {"model_executor/layers/quantization/exl3.py"}
    offenders: list[str] = []

    for source in runtime_root.rglob("*.py"):
        relative = str(source.relative_to(runtime_root))
        if relative in deferred_to_paired_pr:
            continue
        text = source.read_text(encoding="utf-8")
        if any(marker in text for marker in legacy_markers):
            offenders.append(relative)

    assert not offenders, f"legacy SparkInfer package references: {offenders}"
