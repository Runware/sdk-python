from unittest.mock import AsyncMock

import pytest

from runware import RunwareBase, IImageInference, EPromptWeighting


class TestRunwareBase:
    def setup_method(self, method):
        self.runware_base = RunwareBase(api_key="SECRET")
        self.runware_base.connect = AsyncMock()

    @pytest.mark.asyncio
    async def test_image_inference(self):
        self.runware_base._requestImages = AsyncMock()
        request = IImageInference(
            positivePrompt="positive prompt",
            model="test:100@1",
            promptWeighting=EPromptWeighting.sdEmbeds,
        )
        await self.runware_base.imageInference(requestImage=request)

        api_request_object = self.runware_base._requestImages.await_args.kwargs["request_object"]
        assert api_request_object["positivePrompt"] == request.positivePrompt
        assert api_request_object["modelId"] == request.model
        assert api_request_object["promptWeighting"] == request.promptWeighting.value
