import asyncio
import time
from typing import Any, Dict, List, Optional

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..core.types import OperationStatus
from ..exceptions import RunwareOperationError, RunwareTimeoutError
from ..logging_config import get_logger
from ..types import ETaskType, IFrameImage, IVideo, IVideoInference
from ..utils import process_image

logger = get_logger(__name__)


class VideoInferenceOperation(BaseOperation):
    field_mappings = {
        "taskType": "taskType",
        "taskUUID": "taskUUID",
        "status": "status",
        "videoUUID": "videoUUID",
        "videoURL": "videoURL",
        "cost": "cost",
        "seed": "seed",
    }
    response_class = IVideo

    def __init__(self, request: IVideoInference, client=None):
        super().__init__(request.taskUUID, client)
        self.request = request
        self.expected_results = request.numberResults or 1
        self.received_results = 0
        self._processed_videos: Dict[str, Any] = {}
        self._status_monitoring_task: Optional[asyncio.Task] = None
        self._initial_response_received = False
        self._initial_response_event = asyncio.Event()
        self._processed_final_status = False
        self._final_status_lock = asyncio.Lock()
        self._last_server_update = None
        self._consecutive_failed_requests = 0
        self._max_failed_requests = 10

        logger.info(f"Video inference operation {self.operation_id} initialized")
        logger.debug(
            f"Operation {self.operation_id} expects {self.expected_results} results"
        )

    @property
    def operation_type(self) -> str:
        return "videoInference"

    def _setup_message_handlers(self):
        self._message_handlers = {
            "videoInference": self._handle_video_inference,
            "getResponse": self._handle_status_response,
            "error": self._handle_error_message,
        }

    async def _process_video_images(self):
        logger.debug(f"Operation {self.operation_id} processing video images")

        frame_tasks = []
        reference_tasks = []

        try:
            if self.request.frameImages:
                logger.debug(
                    f"Operation {self.operation_id} processing {len(self.request.frameImages)} frame images"
                )
                frame_tasks = [
                    process_image(frame_item.inputImage)
                    for frame_item in self.request.frameImages
                    if isinstance(frame_item, IFrameImage)
                ]

            if self.request.referenceImages:
                logger.debug(
                    f"Operation {self.operation_id} processing {len(self.request.referenceImages)} reference images"
                )
                reference_tasks = [
                    process_image(reference_item)
                    for reference_item in self.request.referenceImages
                ]

            if frame_tasks:
                frame_results = await asyncio.gather(*frame_tasks)
                if frame_results:
                    processed_frame_images = []
                    result_index = 0
                    for frame_item in self.request.frameImages:
                        if isinstance(frame_item, IFrameImage):
                            frame_item.inputImages = frame_results[result_index]
                            result_index += 1
                        processed_frame_images.append(frame_item)
                    self.request.frameImages = processed_frame_images
                    logger.debug(
                        f"Operation {self.operation_id} processed frame images successfully"
                    )

            if reference_tasks:
                reference_results = await asyncio.gather(*reference_tasks)
                if reference_results:
                    self.request.referenceImages = reference_results
                    logger.debug(
                        f"Operation {self.operation_id} processed reference images successfully"
                    )

        except Exception as e:
            logger.error(
                f"Operation {self.operation_id} failed to process video images",
                exc_info=e,
            )
            raise

    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        logger.debug(
            f"Operation {self.operation_id} building video inference request payload"
        )
        await self._process_video_images()

        request_object = {
            "deliveryMethod": self.request.deliveryMethod or "async",
            "taskType": ETaskType.VIDEO_INFERENCE.value,
            "taskUUID": self.operation_id,
            "model": self.request.model,
            "positivePrompt": self.request.positivePrompt.strip(),
            "numberResults": self.expected_results,
        }

        optional_fields = {
            "outputType": self.request.outputType,
            "outputFormat": self.request.outputFormat,
            "outputQuality": self.request.outputQuality,
            "uploadEndpoint": self.request.uploadEndpoint,
            "includeCost": self.request.includeCost,
            "negativePrompt": self.request.negativePrompt,
            "fps": self.request.fps,
            "steps": self.request.steps,
            "seed": self.request.seed,
            "CFGScale": self.request.CFGScale,
            "duration": self.request.duration,
            "width": self.request.width,
            "height": self.request.height,
        }

        for key, value in optional_fields.items():
            if value is not None:
                request_object[key] = value

        # Use ThreadPoolExecutor for serializing frame images
        if self.request.frameImages:
            frame_images_data = []
            for frame_item in self.request.frameImages:
                # Serialize dataclass using ThreadPoolExecutor
                serialized_frame = await cpu_executor.serialize_dataclass(frame_item)
                frame_images_data.append(
                    {k: v for k, v in serialized_frame.items() if v is not None}
                )
            request_object["frameImages"] = frame_images_data

        if self.request.referenceImages:
            request_object["referenceImages"] = self.request.referenceImages

        if self.request.providerSettings:
            provider_dict = self.request.providerSettings.to_request_dict()
            if provider_dict:
                request_object["providerSettings"] = provider_dict

        logger.debug(
            f"Operation {self.operation_id} built video inference request payload"
        )
        return [request_object]

    async def _handle_video_inference(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling video inference message: {message}"
            )

            self._initial_response_received = True
            self._initial_response_event.set()

            # Update last server update timestamp
            self._last_server_update = time.time()

            status = message.get("status")
            logger.info(f"Operation {self.operation_id} received status: {status}")

            # Use lock to prevent duplicate processing of final status
            async with self._final_status_lock:
                if self._processed_final_status:
                    logger.debug(
                        f"Operation {self.operation_id} already processed final status, ignoring"
                    )
                    return

                if status == "success":
                    logger.info(
                        f"Operation {self.operation_id} processing success status"
                    )
                    self._processed_final_status = True
                    await self._stop_status_monitoring()
                    video_data = self._parse_response(message)
                    await self._complete_operation([video_data])
                elif status == "failed":
                    logger.error(
                        f"Operation {self.operation_id} processing failed status"
                    )
                    self._processed_final_status = True
                    await self._stop_status_monitoring()
                    error_message = message.get("error", "Video generation failed")
                    error = RunwareOperationError(
                        f"Video generation failed: {error_message}",
                        operation_id=self.operation_id,
                        operation_type=self.operation_type,
                    )
                    await self._handle_error(error)
                else:
                    # For non-final statuses, start monitoring if not already started
                    await self._start_status_monitoring()

        except Exception as e:
            logger.error(
                f"Error handling video inference message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)

    async def _start_status_monitoring(self):
        if self._status_monitoring_task and not self._status_monitoring_task.done():
            logger.debug(
                f"Operation {self.operation_id} status monitoring already running"
            )
            return

        logger.info(f"Operation {self.operation_id} starting status monitoring")
        self._status_monitoring_task = asyncio.create_task(self._monitor_status())

    async def _stop_status_monitoring(self):
        """Explicitly stop status monitoring task"""
        if self._status_monitoring_task and not self._status_monitoring_task.done():
            logger.info(f"Operation {self.operation_id} stopping status monitoring")
            self._status_monitoring_task.cancel()
            try:
                await asyncio.wait_for(self._status_monitoring_task, timeout=2.0)
                logger.debug(
                    f"Operation {self.operation_id} status monitoring stopped successfully"
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug(
                    f"Operation {self.operation_id} status monitoring stopped (cancelled/timeout)"
                )
            except Exception as e:
                logger.warning(
                    f"Operation {self.operation_id} error stopping status monitoring: {e}"
                )
            finally:
                self._status_monitoring_task = None

    async def _monitor_status(self):
        check_interval = 3.0  # Increased initial interval
        max_interval = 15.0  # Increased max interval
        max_checks = 600  # Increased max checks for longer operations
        check_count = 0

        # Add timeout tracking for server responsiveness
        import time

        no_response_threshold = 120.0  # 2 minutes without server response

        logger.debug(f"Operation {self.operation_id} status monitoring loop started")

        try:
            while check_count < max_checks and self.status == OperationStatus.EXECUTING:
                check_count += 1

                logger.debug(
                    f"Operation {self.operation_id} status check {check_count}/{max_checks}"
                )

                # Check if we haven't received server updates for too long
                current_time = time.time()
                if (
                    self._last_server_update
                    and current_time - self._last_server_update > no_response_threshold
                ):
                    logger.warning(
                        f"Operation {self.operation_id} no server updates for {current_time - self._last_server_update:.1f}s"
                    )

                status_request = {
                    "taskType": ETaskType.GET_RESPONSE.value,
                    "taskUUID": self.operation_id,
                }

                if self.client and self.client.connection_manager:
                    logger.debug(
                        f"Operation {self.operation_id} sending status request"
                    )
                    try:
                        await self.client.connection_manager.send_message(
                            [status_request]
                        )
                        logger.debug(
                            f"Operation {self.operation_id} status request sent successfully"
                        )
                    except Exception as e:
                        logger.error(
                            f"Operation {self.operation_id} failed to send status request: {e}"
                        )
                        self._consecutive_failed_requests += 1

                        if (
                            self._consecutive_failed_requests
                            >= self._max_failed_requests
                        ):
                            logger.error(
                                f"Operation {self.operation_id} too many failed requests, stopping monitoring"
                            )
                            await self._handle_error(
                                RunwareOperationError(
                                    f"Failed to send status requests {self._consecutive_failed_requests} times",
                                    operation_id=self.operation_id,
                                    operation_type=self.operation_type,
                                )
                            )
                            break
                else:
                    logger.error(
                        f"Operation {self.operation_id} no client or connection manager available"
                    )
                    break

                progress = min(check_count / max_checks, 0.9)
                await self._update_progress(progress, "Generating video")

                # Check status again before sleeping to catch quick status changes
                if self.status != OperationStatus.EXECUTING:
                    logger.debug(
                        f"Operation {self.operation_id} status changed to {self.status.value}, stopping monitoring"
                    )
                    break

                try:
                    logger.debug(
                        f"Operation {self.operation_id} sleeping for {check_interval}s"
                    )
                    await asyncio.sleep(check_interval)
                    # Reset failed request counter on successful sleep
                    self._consecutive_failed_requests = 0
                except asyncio.CancelledError:
                    logger.debug(
                        f"Operation {self.operation_id} status monitoring cancelled during sleep"
                    )
                    break

                check_interval = min(
                    check_interval * 1.05, max_interval
                )  # Slower growth

            if self.status == OperationStatus.EXECUTING and check_count >= max_checks:
                logger.error(
                    f"Operation {self.operation_id} timed out after {max_checks} status checks"
                )
                await self._handle_error(
                    RunwareTimeoutError(
                        f"Video generation timed out after {max_checks} status checks",
                        timeout_duration=max_checks * max_interval,
                    )
                )

        except asyncio.CancelledError:
            logger.debug(f"Operation {self.operation_id} status monitoring cancelled")
        except Exception as e:
            logger.error(
                f"Error in status monitoring for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
        finally:
            logger.debug(
                f"Operation {self.operation_id} status monitoring loop finished"
            )

    async def _handle_status_response(self, message: Dict[str, Any]):
        try:
            status = message.get("status")
            logger.info(
                f"Operation {self.operation_id} received status response: {status} (raw message: {message})"
            )

            # Update last server update timestamp
            self._last_server_update = time.time()

            # Reset failed request counter on successful response
            self._consecutive_failed_requests = 0

            # Use same lock and logic as video inference handler
            async with self._final_status_lock:
                if self._processed_final_status:
                    logger.debug(
                        f"Operation {self.operation_id} already processed final status, ignoring status response"
                    )
                    return

                if status == "success":
                    logger.info(
                        f"Operation {self.operation_id} processing success status from getResponse"
                    )
                    self._processed_final_status = True
                    await self._stop_status_monitoring()
                    video_data = self._parse_response(message)
                    await self._complete_operation([video_data])
                elif status == "failed":
                    logger.error(
                        f"Operation {self.operation_id} processing failed status from getResponse"
                    )
                    self._processed_final_status = True
                    await self._stop_status_monitoring()
                    error_message = message.get("error", "Video generation failed")
                    error = RunwareOperationError(
                        f"Video generation failed: {error_message}",
                        operation_id=self.operation_id,
                        operation_type=self.operation_type,
                    )
                    await self._handle_error(error)
                elif status == "pending" or status == "processing":
                    progress_message = message.get("message", f"Status: {status}")
                    current_progress = min(self.progress + 0.1, 0.9)
                    await self._update_progress(current_progress, progress_message)
                    logger.debug(
                        f"Operation {self.operation_id} continuing with status: {status}"
                    )
                else:
                    logger.warning(
                        f"Operation {self.operation_id} unknown status in response: {status}"
                    )

        except Exception as e:
            logger.error(
                f"Error handling status response for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)

    async def _handle_error_message(self, message: Dict[str, Any]):
        # Set initial response event to unblock waiting
        if not self._initial_response_received:
            self._initial_response_received = True
            self._initial_response_event.set()

        # Stop monitoring on error
        await self._stop_status_monitoring()
        return super()._handle_error_message(message)

    async def _cleanup(self):
        """Clean up video inference specific resources"""
        logger.debug(f"Operation {self.operation_id} starting video inference cleanup")

        # Stop status monitoring first
        await self._stop_status_monitoring()

        # Then call parent cleanup
        await super()._cleanup()

        logger.debug(f"Operation {self.operation_id} video inference cleanup completed")
