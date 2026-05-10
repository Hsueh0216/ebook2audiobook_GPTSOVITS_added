import os
from lib.conf import voices_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings

models = {
    "internal": {
        "lang": "multi",
        "repo": None,
        "sub": "",
        "voice": None,
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['GPT-SoVITS']]['samplerate'],
        "api_url": default_engine_settings[TTS_ENGINES['GPT-SoVITS']]['api_url'],
    },
}
