from typing import Any, Dict, List

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..logging_config import get_logger
from ..types import ETaskType, IEnhancedPrompt, IPromptEnhance

logger = get_logger(__name__)


class PromptEnhanceOperation(BaseOperation):
    field_mappings = {
        "taskType": "taskType",
        "taskUUID": "taskUUID",
        "text": "text",
        "cost": "cost",
    }
    response_class = IEnhancedPrompt

    def __init__(self, request: IPromptEnhance, client=None):
        super().__init__(operation_id=None, client=client)
        self.request = request
        self.expected_results = request.promptVersions
        self.received_results = 0
        self._processed_prompts = []

        logger.info(f"Prompt enhance operation {self.operation_id} initialized")
        logger.debug(
            f"Operation {self.operation_id} expects {self.expected_results} prompt versions"
        )

    @property
    def operation_type(self) -> str:
        return "promptEnhance"

    def _setup_message_handlers(self):
        self._message_handlers = {
            "promptEnhance": self._handle_prompt_enhance,
            "error": self._handle_error_message,
        }

    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        task_params = await cpu_executor.serialize_dataclass(
            {
                "taskType": ETaskType.PROMPT_ENHANCE.value,
                "taskUUID": self.operation_id,
                "prompt": self.request.prompt,
                "promptMaxLength": getattr(self.request, "promptMaxLength", 380),
                "promptVersions": self.request.promptVersions,
            }
        )

        if self.request.includeCost:
            task_params["includeCost"] = self.request.includeCost

        return [task_params]

    async def _handle_prompt_enhance(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling prompt enhance message: {message}"
            )
            enhanced_prompt = self._parse_response(message)
            self._processed_prompts.append(enhanced_prompt)
            self.received_results += 1

            progress = min(self.received_results / self.expected_results, 1.0)
            await self._update_progress(progress, partial_results=[enhanced_prompt])

            if self.received_results >= self.expected_results:
                logger.info(
                    f"Operation {self.operation_id} received all expected prompt versions"
                )
                await self._complete_operation()

        except Exception as e:
            logger.error(
                f"Error handling prompt enhance message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
