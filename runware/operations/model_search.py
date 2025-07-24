from dataclasses import fields
from typing import Any, Dict

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..exceptions import RunwareOperationError
from ..logging_config import get_logger
from ..types import ETaskType, IModel, IModelSearch, IModelSearchResponse

logger = get_logger(__name__)


class ModelSearchOperation(BaseOperation):
    field_mappings = {
        "taskUUID": "taskUUID",
        "taskType": "taskType",
        "totalResults": "totalResults",
    }
    response_class = IModelSearchResponse

    def __init__(self, request: IModelSearch, client=None):
        super().__init__(operation_id=None, client=client)
        self.request = request
        logger.info(f"Model search operation {self.operation_id} initialized")

    @property
    def operation_type(self) -> str:
        return "modelSearch"

    async def execute(self) -> IModelSearchResponse:
        results = await super().execute()
        if results is not None:
            return results[0]
        return None

    def _setup_message_handlers(self):
        self._message_handlers = {
            "modelSearch": self._handle_model_search,
            "error": self._handle_error_message,
        }

    async def _build_request_payload(self) -> list[Dict[str, Any]]:
        request_object = await cpu_executor.serialize_dataclass(
            {
                "taskUUID": self.operation_id,
                "taskType": ETaskType.MODEL_SEARCH.value,
            }
        )

        # Add tags if present
        if self.request.tags:
            request_object["tags"] = self.request.tags

        # Add all other fields from payload, excluding additional_params
        for key, value in vars(self.request).items():
            if value is not None and key != "additional_params":
                request_object[key] = value

        return [request_object]

    def _parse_response(self, message: Dict[str, Any]) -> IModelSearchResponse:
        """
        Override base _parse_response to handle complex model search response.
        """
        try:
            # Parse models with additional fields support
            models = []
            for model_data in message.get("results", []):
                model = self._create_model_from_data(model_data)
                models.append(model)

            # Create response with parsed models
            response_data = {
                "results": models,
                "taskUUID": message.get("taskUUID"),
                "taskType": message.get("taskType"),
                "totalResults": message.get("totalResults", 0),
            }

            return IModelSearchResponse(**response_data)
        except Exception as e:
            logger.error(
                f"Operation {self.operation_id} failed to parse model search response",
                exc_info=e,
            )
            raise RunwareOperationError(
                f"Failed to parse model search response: {e}",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )

    def _create_model_from_data(self, model_data: Dict[str, Any]) -> IModel:
        """Create IModel instance from API data with additional fields support."""
        # Get valid fields for IModel
        valid_fields = {f.name for f in fields(IModel)}

        # Separate known and unknown fields
        known_fields = {}
        additional_fields = {}

        for key, value in model_data.items():
            if key in valid_fields:
                known_fields[key] = value
            else:
                additional_fields[key] = value

        # Add additional_fields if there are any unknown fields
        if additional_fields:
            known_fields["additional_fields"] = additional_fields

        return IModel(**known_fields)

    async def _handle_model_search(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling model search message: {message}"
            )

            if message.get("error") or message.get("code"):
                error_message = message.get(
                    "message", message.get("error", "Unknown error")
                )
                error_code = message.get("code")

                error = RunwareOperationError(
                    f"Model search error: {error_message}",
                    operation_id=self.operation_id,
                    operation_type=self.operation_type,
                    code=error_code,
                )
                await self._handle_error(error)
                return

            # Parse and complete the operation using base _parse_response
            model_search_response = self._parse_response(message)
            await self._complete_operation([model_search_response])

        except Exception as e:
            logger.error(
                f"Error handling model search message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
