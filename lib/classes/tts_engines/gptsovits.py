"""
GPT-SoVITS TTS Engine for ebook2audiobook

Integrates GPT-SoVITS via its v2 API (api_v2.py).
Requires a running GPT-SoVITS WebUI instance with --api flag enabled.

Usage:
  1. Start GPT-SoVITS: python api_v2.py -a 127.0.0.1 -p 9880
  2. Run ebook2audiobook: app.py --tts_engine gptsovits --voice /path/to/ref.wav
"""

import os, requests, json, tempfile, warnings

from typing import Any
from multiprocessing.managers import DictProxy
from pathlib import Path

from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.conf import tts_dir, voices_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings


class GPTSovits(TTSUtils, TTSRegistry, name='gptsovits'):

    def __init__(self, session: DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.audio_segments = []

            # Load preset config
            self.models = load_engine_presets(self.session['tts_engine'])
            model_cfg = self.models.get(self.session.get('fine_tuned', 'internal'), {})
            self.params = {
                'samplerate': model_cfg.get('samplerate', 24000),
            }

            # API connection settings
            self.api_url = self.session.get('gptsovits_api_url', 'http://127.0.0.1:9880')
            self.api_url = self.api_url.rstrip('/')

            # Reference audio voice settings
            self.current_ref_audio = None
            self.current_prompt_text = self.session.get('gptsovits_prompt_text', '')
            self.current_prompt_lang = self.session.get('gptsovits_prompt_lang',
                                                         self._map_language(self.session.get('language_iso1', 'zh')))

            self.device = devices['CUDA']['proc'] if self.session['device'] in [
                devices['CUDA']['proc'], devices['ROCM']['proc'], devices['JETSON']['proc']
            ] else self.session['device']

            # Verify API is reachable on init
            if not self._check_api_health():
                warnings.warn(
                    f'GPT-SoVITS API not reachable at {self.api_url}. '
                    f'Make sure api_v2.py is running with --api flag.'
                )

        except Exception as e:
            raise ValueError(f'GPTSovits.__init__() error: {e}')

    # ──────────────────────────────────────
    # Language mapping
    # ──────────────────────────────────────

    @staticmethod
    def _map_language(iso_code: str) -> str:
        """Map ISO language codes to GPT-SoVITS language names."""
        mapping = {
            'zho': 'zh', 'zh-cn': 'zh', 'zh-tw': 'zh',
            'eng': 'en', 'en': 'en',
            'jpn': 'ja', 'ja': 'ja',
            'kor': 'ko', 'ko': 'ko',
            'yue': 'yue',
        }
        return mapping.get(iso_code, iso_code)

    # ──────────────────────────────────────
    # API health check
    # ──────────────────────────────────────

    def _check_api_health(self) -> bool:
        """Verify the GPT-SoVITS API is responding."""
        try:
            resp = requests.get(f'{self.api_url}/control?command=restart', timeout=3)
            return resp.status_code < 500
        except (requests.ConnectionError, requests.Timeout):
            return False

    # ──────────────────────────────────────
    # Voice selection
    # ──────────────────────────────────────

    def _set_voice(self, block_voice: str | None) -> tuple:
        """
        Set the reference audio for voice cloning.
        block_voice: path to a reference audio file (.wav)
        Returns (voice_path, error)
        """
        try:
            if block_voice is None:
                block_voice = self.session.get('voice')
            if block_voice is None:
                return None, 'No voice file specified. Use --voice to provide a reference audio path.'

            if not os.path.isfile(block_voice):
                return None, f'Reference audio file not found: {block_voice}'

            self.current_ref_audio = block_voice
            return block_voice, None
        except Exception as e:
            return None, f'_set_voice() error: {e}'

    # ──────────────────────────────────────
    # Engine loading (no-op for API mode)
    # ──────────────────────────────────────

    def load_engine(self) -> bool:
        """
        Verify the GPT-SoVITS API is available.
        Returns True if reachable, False otherwise.
        """
        if self._check_api_health():
            msg = f'GPT-SoVITS API connected at {self.api_url}'
            print(msg)
            return True
        msg = (
            f'WARNING: GPT-SoVITS API not reachable at {self.api_url}.\n'
            f'Make sure to start api_v2.py first:\n'
            f'  python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml\n'
            f'Conversion will fail if the API is not running.'
        )
        print(msg)
        return False

    # ──────────────────────────────────────
    # Core conversion
    # ──────────────────────────────────────

    def convert(self, sentence_file: str, sentence: str, **kwargs) -> tuple:
        """
        Convert a sentence to audio using GPT-SoVITS API.

        Args:
            sentence_file: Path where the output audio will be saved.
            sentence: Text to synthesize.
            **kwargs: May contain 'block_voice' for per-block voice override.

        Returns:
            (True, None) on success, (False, error_message) on failure.
        """
        try:
            import torch
            import numpy as np

            if not self._check_api_health():
                return False, (
                    f'GPT-SoVITS API is not running at {self.api_url}.\n'
                    f'Start it with: python api_v2.py -a 127.0.0.1 -p 9880'
                )

            # Process SML tags (break, pause, voice switching)
            sentence_parts = self._split_sentence_on_sml(sentence)
            self.params['block_voice'] = kwargs.get('block_voice', self.session.get('voice'))

            # Set inline voice if specified via SML tags
            if self.params.get('inline_voice'):
                self.params['current_voice'] = self.params['inline_voice']
            else:
                self.params['current_voice'], error = self._set_voice(self.params['block_voice'])
                if self.params['current_voice'] is None and error is not None:
                    return False, error

            self.audio_segments = []

            for part in sentence_parts:
                part = part.strip()
                if not part:
                    continue

                # Handle SML-only tags (break/pause/voice)
                if SML_TAG_PATTERN.fullmatch(part):
                    success, error = self._convert_sml(part)
                    if not success:
                        return False, error
                    continue

                # Skip parts with no alphanumeric content
                if not any(c.isalnum() for c in part):
                    continue

                # Call GPT-SoVITS API
                success, audio_data = self._api_infer(part)
                if not success:
                    return False, audio_data

                # Convert response to tensor
                try:
                    import soundfile as sf
                    import io

                    # Read WAV bytes from API response
                    audio_array, sr = sf.read(io.BytesIO(audio_data))
                    # Resample if needed
                    if sr != self.params['samplerate']:
                        import torchaudio
                        tensor = torch.from_numpy(audio_array).float()
                        resampler = torchaudio.transforms.Resample(sr, self.params['samplerate'])
                        tensor = resampler(tensor)
                    else:
                        tensor = torch.from_numpy(audio_array).float()

                    # Ensure 1D tensor (mono)
                    if tensor.dim() > 1:
                        tensor = tensor.mean(dim=0)

                    # Add channel dimension for torchaudio
                    part_tensor = tensor.unsqueeze(0).cpu()

                    if part_tensor.numel() > 0:
                        self.audio_segments.append(part_tensor)
                        # Add pause between sentences
                        silence_time = int(np.random.uniform(0.3, 0.6) * 100) / 100
                        break_tensor = torch.zeros(1, int(self.params['samplerate'] * silence_time))
                        self.audio_segments.append(break_tensor.clone())
                    else:
                        return False, 'Generated audio tensor is empty'

                except Exception as e:
                    return False, f'Failed to process audio response: {e}'

            # Concatenate all segments and save
            if self.audio_segments:
                segment_tensor = torch.cat(self.audio_segments, dim=-1)
                if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                    return False, f'audio_save() error: cannot save {sentence_file}'
                del segment_tensor
                self.cleanup_memory()
                self.audio_segments = []
                if not os.path.exists(sentence_file):
                    return False, f'Cannot create {sentence_file}'
                return True, None
            else:
                return False, 'No audio segments generated'

        except Exception as e:
            self.cleanup_memory()
            return False, self.log_exception(f'{self.__class__.__name__}.convert()', e)

    # ──────────────────────────────────────
    # GPT-SoVITS API inference
    # ──────────────────────────────────────

    def _api_infer(self, text: str) -> tuple:
        """
        Send text to GPT-SoVITS API for inference.

        Args:
            text: Text to synthesize.

        Returns:
            (True, wav_bytes) on success, (False, error_message) on failure.
        """
        try:
            # Build inference parameters
            text_lang = self._map_language(self.session.get('language_iso1', 'zh'))

            params = {
                'text': text,
                'text_lang': text_lang,
                'ref_audio_path': self.params.get('current_voice', ''),
                'prompt_lang': self.current_prompt_lang,
                'media_type': 'wav',
                'streaming_mode': False,
            }

            # Add optional prompt text if available
            if self.current_prompt_text:
                params['prompt_text'] = self.current_prompt_text

            # Add inference hyperparameters from session config
            for key, default_val in [
                ('gptsovits_top_k', 15),
                ('gptsovits_top_p', 1.0),
                ('gptsovits_temperature', 1.0),
                ('gptsovits_speed_factor', 1.0),
                ('gptsovits_repetition_penalty', 1.35),
                ('gptsovits_seed', -1),
            ]:
                val = self.session.get(key, default_val)
                if val is not None:
                    param_key = key.removeprefix('gptsovits_')
                    params[param_key] = val

            # Make API request
            resp = requests.post(
                f'{self.api_url}/tts',
                json=params,
                timeout=120,
                headers={'Content-Type': 'application/json'},
            )

            if resp.status_code == 200:
                return True, resp.content
            else:
                try:
                    err_detail = resp.json()
                except (json.JSONDecodeError, ValueError):
                    err_detail = resp.text[:500]
                return False, f'GPT-SoVITS API error (HTTP {resp.status_code}): {err_detail}'

        except requests.ConnectionError:
            return False, (
                f'Cannot connect to GPT-SoVITS API at {self.api_url}.\n'
                f'Make sure api_v2.py is running.'
            )
        except requests.Timeout:
            return False, 'GPT-SoVITS API request timed out (120s).'
        except Exception as e:
            return False, f'_api_infer() error: {e}'

    # ──────────────────────────────────────
    # VTT file creation
    # ──────────────────────────────────────

    def create_vtt(self, all_sentences: list) -> bool:
        if self._build_vtt_file(all_sentences):
            return True
        return False
