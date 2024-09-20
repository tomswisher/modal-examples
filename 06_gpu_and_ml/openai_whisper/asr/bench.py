import pytest
import requests
import json
import time

# Define URL and audio files
URL = 'https://modal-labs--latency-test-web.modal.run'

AUDIO_FILES = {
    'thirty': 'wavs/thirty.wav',
    'short': 'wavs/short.wav',
    'preamble': 'wavs/preamble.wav',
    'long': 'wavs/long.wav'
}

SCENARIOS = [  # (audio_type, batch_size, assisted)
    # ('short', '24', 'false'),
    # ('short', '1', 'true'),
    # ('long', '24', 'false'),
    # ('long', '1', 'true'),
    # ('preamble', '24', 'false'),
    # ('preamble', '1', 'true'),
    # ('short', '24', 'false'),
    # ('preamble', '1', 'false'),
    # ('long', '24', 'false'),
    # ('long', '1', 'false'),
    # ('preamble', '24', 'false'),
    # ('preamble', '1', 'true'),
    ('thirty', '1', 'false')
]
        



# Fixture to perform HTTP POST request
@pytest.fixture
def perform_request(benchmark):
    def _perform_request(audio_type, batch_size, assisted, bench=True):
        session = requests.Session()
        for _ in range(5):
            print(f"Starting request at {time.monotonic()}")
            session.post(URL, files = {'file': (AUDIO_FILES['long'], open(AUDIO_FILES['long'], 'rb'), 'audio/wav')}, data=None)
        
        def fn():
            files = {'file': (AUDIO_FILES[audio_type], open(AUDIO_FILES[audio_type], 'rb'), 'audio/wav')}
            parameters = {'batch_size': batch_size, 'assisted': assisted}
            data = {'parameters': json.dumps(parameters)}

            return session.post(URL, files=files, data=data)
        if bench:
            return benchmark(fn)
        else:
            return fn()

    return _perform_request

# Benchmark test function
@pytest.mark.parametrize("audio_type, batch_size, assisted", SCENARIOS)
def test_bench(perform_request, audio_type, batch_size, assisted):
    result = perform_request(audio_type, batch_size, assisted)
    assert result.status_code == 200  # Assert successful response

# @pytest.mark.parametrize("audio_type, batch_size, assisted", SCENARIOS)
# def test_singleflight(perform_request, audio_type, batch_size, assisted):
#     result = perform_request(audio_type, batch_size, assisted, bench=False)
#     assert result.status_code == 200  # Assert successful response
#     print(json.dumps(result.json(), indent=4))  # print response

# curl -X POST 'https://irfansharif--diarization-model-web.modal.run' \
#     -F 'file=@wavs/short.wav;type=audio/wav' \
#     -F 'parameters={"batch_size":24,"assisted":false}'


# if __name__ == "__main__":
#     main()
