from pathlib import Path


def test_runtime_does_not_reference_legacy_sparkinfer_package() -> None:
    runtime_root = Path(__file__).parents[1] / "vllm"
    legacy_markers = ("sparkinfer.", "SPARKINFER_")
    offenders: list[str] = []

    for source in runtime_root.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        if any(marker in text for marker in legacy_markers):
            offenders.append(str(source.relative_to(runtime_root)))

    assert not offenders, f"legacy SparkInfer package references: {offenders}"
