import logging
from typing import Any

from ..settings import settings
from ..constants.constants import *

logger = logging.getLogger(__name__)


class GrokAPI:
    def __init__(self, api_key: str, model: str = None, temperature: float = None):
        self.api_key = api_key
        self.model = model or settings.model_name
        self.temperature = temperature or settings.temperature
        self.base_url = XAI_BASE_URL

    def invoke(self, prompt: Any) -> dict[str, Any]:
        try:
            import requests

            if hasattr(prompt, "to_string"):
                prompt_content = prompt.to_string()
            elif hasattr(prompt, "text"):
                prompt_content = prompt.text
            elif hasattr(prompt, "content"):
                prompt_content = prompt.content
            else:
                prompt_content = str(prompt)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": CONTENT_TYPE_JSON,
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_content}],
                "temperature": self.temperature,
            }

            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=data
            )

            if response.status_code == HTTP_STATUS_OK:
                result = response.json()
                return {"content": result["choices"][XAI_FIRST_CHOICE_INDEX]["message"]["content"], "success": True}
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {"content": f"API Error: {response.status_code}", "success": False}

        except Exception as e:
            logger.error(f"Failed to call Grok API: {e}")
            return {"content": f"Error: {str(e)}", "success": False}

    def bind(self, **kwargs: Any) -> "GrokAPI":
        return self

    def __call__(self, prompt: Any) -> str:
        result = self.invoke(prompt)
        content = result.get("content", "Error: No content returned")
        return str(content)


class LLMFactory:

    @staticmethod
    def create_llm() -> "GrokAPI":
        return GrokAPI(api_key=settings.xai_api_key)


def create_llm() -> "GrokAPI":
    return LLMFactory.create_llm()
