import os
import requests
import io
import urllib.parse
from typing import Optional, Dict, Any, List
from .base import ASRModel
from ..core.audio_utils import convert_to_wav_bytes


class Qwen3Model(ASRModel):
    """Qwen3 ASR model using external API."""
    
    def __init__(self, endpoint: str = None, api_key: str = None):
        """
        Initialize the Qwen3 API model.
        
        Args:
            endpoint: API endpoint URL (default from environment)
            api_key: API key (default from environment)
        """
        self.endpoint = endpoint or os.getenv("QWEN3_ENDPOINT", "http://localhost:8005/asr")
        self.api_key = api_key or os.getenv("QWEN3_API_KEY", "AIRRVie_api_key")
        self.supported_languages = self._load_supported_languages()
        
    def _load_supported_languages(self) -> List[str]:
        """Load supported languages for Qwen3 model."""
        common_languages = [
            "eng_Latn",
            "vie_Latn",
            "fra_Latn",
            "spa_Latn",
            "deu_Latn",
            "ita_Latn",
            "por_Latn",
            "rus_Cyrl",
            "jpn_Jpan",
            "kor_Hang",
            "cmn_Hans",
            "cmn_Hant",
            "ara_Arab",
            "hin_Deva",
        ]
        return common_languages
    
    def transcribe(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio bytes to text using Qwen3 API.
        
        Args:
            audio_bytes: Audio data in bytes
            task: Only "transcribe" is supported
            language: Language code in format {language_code}_{script} (e.g., "vie_Latn")
            **kwargs: Additional parameters (ignored for API)
            
        Returns:
            Transcribed text
        """
        if task != "transcribe":
            raise ValueError("Qwen3 API only supports transcription, not translation")
        
        try:
            wav_bytes = convert_to_wav_bytes(audio_bytes)
        except Exception as e:
            raise Exception(f"Audio format conversion failed: {str(e)}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        import time
        audio_stream = io.BytesIO(wav_bytes)
        audio_stream.seek(0)
        
        timestamp = int(time.time() * 1000)
        filename = f"audio_{timestamp}.wav"
        
        files = {
            "audio": (filename, audio_stream, "audio/wav")
        }

        def normalize_lang_code(lang: Optional[str]) -> Optional[str]:
            if not lang:
                return None
            lang = (lang or "").strip()
            if lang == "":
                return None

            iso2_to_iso3 = {
                "en": "eng",
                "vi": "vie",
                "fr": "fra",
                "de": "deu",
                "es": "spa",
                "it": "ita",
                "pt": "por",
                "ru": "rus",
                "ja": "jpn",
                "ko": "kor",
                "zh": "cmn",
                "ar": "ara",
                "hi": "hin",
            }

            lower = lang.lower()
            if lower in iso2_to_iso3:
                return iso2_to_iso3[lower]

            # If given something like "vie_Latn" or "rus_Cyrl", drop the script.
            if "_" in lang:
                return lang.split("_", 1)[0]

            return lang

        def candidate_lang_codes(lang: Optional[str]) -> list[str]:
            """Return candidate lang_code values to try with Qwen upstream.

            Some deployments expect ISO-ish codes (e.g. 'vie', 'eng'), while others
            expect English names (e.g. 'Vietnamese', 'English').
            """
            normalized = normalize_lang_code(lang)
            if not normalized:
                return []

            candidates: list[str] = []
            if normalized not in candidates:
                candidates.append(normalized)

            name_map = {
                "eng": "English",
                "en": "English",
                "vie": "Vietnamese",
                "vi": "Vietnamese",
            }

            lower = normalized.lower()
            if lower in name_map and name_map[lower] not in candidates:
                candidates.append(name_map[lower])

            # If user already passed a name like "Vietnamese", keep it as well.
            if normalized not in candidates:
                candidates.append(normalized)

            return candidates

        lang_candidates = candidate_lang_codes(language)
        data = {}
        if lang_candidates:
            data["lang_code"] = lang_candidates[0]

        import hashlib
        audio_hash = hashlib.md5(wav_bytes).hexdigest()[:8]
        
        print(f"[DEBUG] Calling Qwen3 API: {self.endpoint}")
        print(f"[DEBUG] Language: {data.get('lang_code')}")
        print(f"[DEBUG] Original audio size: {len(audio_bytes)} bytes")
        print(f"[DEBUG] WAV audio size: {len(wav_bytes)} bytes")
        print(f"[DEBUG] Audio hash: {audio_hash}")
        print(f"[DEBUG] API Key present: {'Yes' if self.api_key else 'No'}")
        print(f"[DEBUG] Using filename: {filename}")
        
        try:
            def _add_endpoint(candidates: list, url: str):
                if not url:
                    return
                if url not in candidates:
                    candidates.append(url)

            candidates: list[str] = []
            endpoint = (self.endpoint or "").strip()
            _add_endpoint(candidates, endpoint)

            base = endpoint.rstrip("/")
            if base:
                _add_endpoint(candidates, f"{base}/asr")
                _add_endpoint(candidates, f"{base}/transcribe")
                _add_endpoint(candidates, f"{base}/transcribe/upload")

            if endpoint:
                parsed = urllib.parse.urlparse(endpoint)
                path = parsed.path or ""
                if "/asr_q3_1_7B" in path:
                    direct_base = urllib.parse.urlunparse(parsed._replace(netloc=f"{parsed.hostname}:8005", path="", params="", query="", fragment=""))
                    direct_base = direct_base.rstrip("/")
                    _add_endpoint(candidates, f"{direct_base}/asr")
                    _add_endpoint(candidates, f"{direct_base}/transcribe")
                if "/asr_q3_0_6B" in path:
                    direct_base = urllib.parse.urlunparse(parsed._replace(netloc=f"{parsed.hostname}:8006", path="", params="", query="", fragment=""))
                    direct_base = direct_base.rstrip("/")
                    _add_endpoint(candidates, f"{direct_base}/asr")
                    _add_endpoint(candidates, f"{direct_base}/transcribe")

            response = None
            last_error = None
            # Try different lang_code values first (if provided), then try endpoint fallbacks.
            all_langs = lang_candidates if lang_candidates else [None]

            for lang_code in all_langs:
                if lang_code:
                    data["lang_code"] = lang_code
                elif "lang_code" in data:
                    data.pop("lang_code", None)

                for url in candidates:
                    print(f"[DEBUG] POST {url}")
                    response = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=120,
                    )

                    print(f"[DEBUG] Response status: {response.status_code}")
                    print(f"[DEBUG] Response headers: {response.headers}")

                    if response.status_code == 200:
                        break
                    if response.status_code in {400, 404, 405, 422}:
                        last_error = response.text
                        continue

                    print(f"[DEBUG] Response error: {response.text}")
                    raise Exception(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                if response is not None and response.status_code == 200:
                    break

            if response is None:
                raise Exception("No endpoint configured")

            if response.status_code != 200:
                print(f"[DEBUG] Response error: {response.text}")
                if last_error is not None:
                    raise Exception(f"API request failed with status {response.status_code}: {last_error}")
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            print(f"[DEBUG] Response JSON: {result}")
            
            if 'text' in result:
                text = result['text'].strip()
                print(f"[DEBUG] Extracted text: {text}")
                return text
            else:
                print(f"[DEBUG] No 'text' field in response. Available keys: {result.keys()}")
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 0:
                        print(f"[DEBUG] Using alternative field '{key}': {value}")
                        return value.strip()
                raise Exception(f"No text field found in response: {result}")
            
        except Exception as e:
            print(f"[DEBUG] Exception in transcribe: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    def get_available_languages(self) -> list:
        """Get list of supported language codes."""
        return self.supported_languages
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Qwen3 ASR",
            "endpoint": self.endpoint,
            "supported_languages_count": len(self.supported_languages),
            "supported_languages": self.supported_languages[:10],
            "task": "transcribe",
            "provider": "Qwen3 API"
        }
