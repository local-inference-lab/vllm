from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import RepetitionDetectionParams


def test_repetition_detection_reaches_sampling_params() -> None:
    request = ResponsesRequest(
        model="test-model",
        input="hello",
        repetition_detection={
            "min_pattern_size": 8,
            "max_pattern_size": 32,
            "min_count": 4,
        },
    )

    params = request.to_sampling_params(default_max_tokens=128)

    assert params.repetition_detection == RepetitionDetectionParams(
        min_pattern_size=8,
        max_pattern_size=32,
        min_count=4,
    )
