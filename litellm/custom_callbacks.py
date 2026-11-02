from typing import Literal

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth


class MyCustomHandler(CustomLogger):

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ):
        model = data.get("model", "").lower()

        if "qwen" in model:
            print("[CALLBACK PROCESSING]: qwen", flush=True)

            messages = data.get("messages", [])

            # Only convert mid-conversation system messages when
            # Anthropic provides a top-level system prompt.
            if data.get("system"):
                for message in messages:
                    if message.get("role") == "system":
                        message["role"] = "user"

            return data
        return data


proxy_handler_instance = MyCustomHandler()
