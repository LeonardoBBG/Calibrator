import json
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_REQUEST_MAX_RETRIES = 1
DEFAULT_REQUEST_RETRY_DELAY_SECONDS = 5

class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        require_temperature_support: bool = True,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = False,
        request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
        request_retry_delay_seconds: int = DEFAULT_REQUEST_RETRY_DELAY_SECONDS
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.require_temperature_support = require_temperature_support
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.request_timeout_seconds = request_timeout_seconds
        self.request_max_retries = request_max_retries
        self.request_retry_delay_seconds = request_retry_delay_seconds

    def complete_json(self, system_prompt: str, user_payload: Dict) -> Dict:
        """Send prompt and payload to LLM and return parsed JSON response."""
        cache_path = self._cache_path(system_prompt, user_payload)
        if cache_path is not None and cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        if self.provider == "openai":
            response = self._openai_complete(system_prompt, user_payload)
        elif self.provider == "anthropic":
            response = self._anthropic_complete(system_prompt, user_payload)
        elif self.provider == "local":
            response = self._local_complete(system_prompt, user_payload)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
        return response

    def _cache_path(self, system_prompt: str, user_payload: Dict) -> Optional[Path]:
        if not self.cache_enabled or self.cache_dir is None:
            return None
        cache_input = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "require_temperature_support": self.require_temperature_support,
            "max_tokens": self.max_tokens,
            "system_prompt": system_prompt,
            "user_payload": user_payload
        }
        encoded = json.dumps(cache_input, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        digest = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _openai_complete(self, system_prompt: str, user_payload: Dict) -> Dict:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in your environment or in the "
                "notebook Model/API Configuration cell before RUN_LLM=True."
            )

        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\nReturn valid JSON only."
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, separators=(',', ':'))
                }
            ],
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "store": False
        }
        if self.require_temperature_support:
            request_body["temperature"] = self.temperature
        response = self._post_openai_chat_completion(request_body, api_key)

        try:
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenAI response shape: {response}") from exc
        if finish_reason == "length":
            raise RuntimeError(
                "OpenAI stopped because the response hit the token limit before valid JSON was complete. "
                "Increase max_completion_tokens/config.max_tokens or reduce the requested output size."
            )
        return self._parse_json_content(content)

    def _post_openai_chat_completion(self, request_body: Dict, api_key: str) -> Dict:
        try:
            return self._post_json(
                "https://api.openai.com/v1/chat/completions",
                request_body,
                {"Authorization": f"Bearer {api_key}"}
            )
        except RuntimeError as exc:
            message = str(exc)
            if "max_completion_tokens" in message and "Unsupported parameter" in message:
                fallback_body = dict(request_body)
                fallback_body["max_tokens"] = fallback_body.pop("max_completion_tokens")
                return self._post_json(
                    "https://api.openai.com/v1/chat/completions",
                    fallback_body,
                    {"Authorization": f"Bearer {api_key}"}
                )
            if "temperature" in message and "Unsupported value" in message:
                if self.require_temperature_support:
                    raise RuntimeError(
                        f"OpenAI model '{self.model}' does not support configured temperature "
                        f"{self.temperature}. For this calibration task, choose a model that "
                        "supports low temperature, or set REQUIRE_TEMPERATURE_SUPPORT=False "
                        "in the notebook to allow the model default."
                    ) from exc
                fallback_body = dict(request_body)
                fallback_body.pop("temperature", None)
                return self._post_json(
                    "https://api.openai.com/v1/chat/completions",
                    fallback_body,
                    {"Authorization": f"Bearer {api_key}"}
                )
            raise

    def _anthropic_complete(self, system_prompt: str, user_payload: Dict) -> Dict:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment or in the "
                "notebook Model/API Configuration cell before RUN_LLM=True."
            )
        raise NotImplementedError(
            "Anthropic provider is selected but the API call is not implemented yet. "
            "Use the notebook Model/API Configuration cell to choose a provider/model, "
            "then implement _anthropic_complete for that provider in src/llm_client.py."
        )

    def _local_complete(self, system_prompt: str, user_payload: Dict) -> Dict:
        raise NotImplementedError(
            "Local provider is selected but the local model call is not implemented yet. "
            "Use the notebook Model/API Configuration cell to choose a provider/model, "
            "then implement _local_complete in src/llm_client.py."
        )

    def _post_json(self, url: str, body: Dict, headers: Dict[str, str]) -> Dict:
        payload = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        request_headers = {
            "Content-Type": "application/json",
            **headers
        }
        request = Request(url, data=payload, headers=request_headers, method="POST")
        response_body = None
        attempts = max(1, self.request_max_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=self.request_timeout_seconds) as response:
                    response_body = response.read().decode('utf-8')
                break
            except HTTPError as exc:
                error_body = exc.read().decode('utf-8', errors='replace')
                if exc.code == 400 and (
                    "invalid model ID" in error_body
                    or "does not exist" in error_body
                    or "you do not have access" in error_body
                ):
                    raise RuntimeError(
                        f"OpenAI rejected model '{self.model}' as an invalid or inaccessible model ID. "
                        "Set MODEL_NAME in the first notebook cell to a valid OpenAI model "
                        "that supports Chat Completions, such as 'gpt-4.1-mini'."
                    ) from exc
                raise RuntimeError(f"API request failed with HTTP {exc.code}: {error_body}") from exc
            except URLError as exc:
                raise RuntimeError(f"API request failed: {exc.reason}") from exc
            except TimeoutError as exc:
                if attempt < attempts:
                    time.sleep(self.request_retry_delay_seconds)
                    continue
                raise RuntimeError(
                    f"API request timed out after {self.request_timeout_seconds} seconds "
                    f"({attempts} attempt(s)). For large calibration inputs, reduce input size, "
                    "use a faster model, or increase REQUEST_TIMEOUT_SECONDS in the notebook/config."
                ) from exc
        return json.loads(response_body)

    def _parse_json_content(self, content: str) -> Dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            stripped = content.strip()
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                return json.loads("\n".join(lines))
            preview = content[-500:] if content else ""
            raise ValueError(
                "Model response was not valid complete JSON. "
                f"Last 500 characters received: {preview}"
            )
