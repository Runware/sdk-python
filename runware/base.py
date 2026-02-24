import asyncio
import inspect
import logging
import os
import re
from asyncio import gather
from dataclasses import asdict
from random import uniform
from typing import List, Optional, Union, Callable, Any, Dict, Tuple

from websockets.protocol import State

from .logging_config import configure_logging

from .async_retry import asyncRetry
from .reconnection import ConnectionState, ReconnectionManager
from .types import (
    Environment,
    IImageInference,
    IPhotoMaker,
    IImageCaption,
    IImageToText,
    IImageBackgroundRemoval,
    ISafety,
    IPromptEnhance,
    IEnhancedPrompt,
    IImageUpscale,
    IUploadModelBaseType,
    IUploadModelResponse,
    ReconnectingWebsocketProps,
    UploadImageType,
    MediaStorageType,
    EPreProcessorGroup,
    File,
    ETaskType,
    EDeliveryMethod,
    IModelSearch,
    IModelSearchResponse,
    IControlNet,
    IVideo,
    IVideoCaption,
    IVideoToText,
    IVideoBackgroundRemoval,
    IVideoUpscale,
    IVideoInference,
    IVideoAdvancedFeatures,
    IAcceleratorOptions,
    IAudio,
    IAudioInference,
    IFrameImage,
    IAsyncTaskResponse,
    IVectorize,
    OperationState,
    I3dInference,
    I3d,
    IGetResponseRequest,
    IUploadImageRequest,
    IUploadMediaRequest,
    ITextInference,
    IText,
)
from .types import IImage, IError, SdkType, ListenerType
from .utils import (
    BASE_RUNWARE_URLS,
    getUUID,
    fileToBase64,
    createImageFromResponse,
    createImageToTextFromResponse,
    createVideoToTextFromResponse,
    createEnhancedPromptsFromResponse,
    instantiateDataclassList,
    RunwareAPIError,
    RunwareError,
    instantiateDataclass,
    TIMEOUT_DURATION,
    accessDeepObject,
    getIntervalWithPromise,
    removeListener,
    LISTEN_TO_IMAGES_KEY,
    isLocalFile,
    process_image,
    createAsyncTaskResponse,
    VIDEO_INITIAL_TIMEOUT,
    TEXT_INITIAL_TIMEOUT,
    VIDEO_POLLING_DELAY,
    WEBHOOK_TIMEOUT,
    IMAGE_INFERENCE_TIMEOUT,
    IMAGE_OPERATION_TIMEOUT,
    PROMPT_ENHANCE_TIMEOUT,
    IMAGE_UPLOAD_TIMEOUT,
    MODEL_UPLOAD_TIMEOUT,
    IMAGE_INITIAL_TIMEOUT,
    IMAGE_POLLING_DELAY,
    TEXT_POLLING_DELAY,
    AUDIO_INITIAL_TIMEOUT,
    AUDIO_INFERENCE_TIMEOUT,
    AUDIO_POLLING_DELAY,
    MAX_POLLS,
    MAX_POLLS_VIDEO_GENERATION,
    MAX_POLLS_AUDIO_GENERATION,
    MAX_POLLS_3D_GENERATION,
    MAX_POLLS_IMAGE_GENERATION,
    MAX_RETRY_ATTEMPTS,
    MAX_CONCURRENT_REQUESTS,
)

# Configure logging
configure_logging(log_level=logging.CRITICAL)

logger = logging.getLogger(__name__)


class RunwareBase:
    def __init__(
            self,
            api_key: str,
            url: str = BASE_RUNWARE_URLS[Environment.PRODUCTION],
            timeout: int = TIMEOUT_DURATION,
            log_level=logging.CRITICAL,
    ):
        if timeout <= 0:
            raise ValueError("Timeout must be greater than 0 milliseconds")

        # Configure logging
        configure_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self._ws: Optional[ReconnectingWebsocketProps] = None
        self._listeners: List[ListenerType] = []
        self._apiKey: str = api_key
        self._url: Optional[str] = url
        self._timeout: int = timeout
        self._globalMessages: Dict[str, Any] = {}
        self._globalImages: List[IImage] = []
        self._globalError: Optional[IError] = None
        self._connectionSessionUUID: Optional[str] = None
        self._invalidAPIkey: Optional[str] = None
        self._sdkType: SdkType = SdkType.SERVER
        self._messages_lock = asyncio.Lock()
        self._images_lock = asyncio.Lock()
        self._listener_tasks = set()
        self._reconnection_manager = ReconnectionManager(logger=self.logger)
        self._reconnect_lock = asyncio.Lock()
        self._pending_operations: Dict[str, Dict[str, Any]] = {}
        self._operations_lock = asyncio.Lock()
        if MAX_CONCURRENT_REQUESTS <= 0:
            raise ValueError(
                "RUNWARE_MAX_CONCURRENT_REQUESTS must be greater than 0 "
                f"(got {MAX_CONCURRENT_REQUESTS}). A value of 0 would cause asyncio.Semaphore to deadlock."
            )
        self._request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _register_pending_operation(
            self,
            task_uuid: str,
            expected_results: int = 1,
            on_partial: "Optional[Callable[[List[Any], Optional[IError]], None]]" = None,
            complete_predicate: "Optional[Callable[[Dict[str, Any]], bool]]" = None,
            result_filter: "Optional[Callable[[Dict[str, Any]], bool]]" = None
    ) -> "Tuple[asyncio.Future, bool]":
        async with self._operations_lock:
            if task_uuid in self._pending_operations:
                existing_op = self._pending_operations[task_uuid]
                existing_state = existing_op.get("state")
                existing_future = existing_op.get("future")

                if existing_state == OperationState.REGISTERED:
                    duplicated_exc_body = {
                        "code": "conflictTaskUUID",
                        "message": f"Task {task_uuid} already has a pending request. "
                                   f"Use unique taskUUID for each concurrent request."
                    }
                    if existing_future and not existing_future.done():
                        existing_future.set_exception(RunwareAPIError(duplicated_exc_body))
                    raise RunwareAPIError(duplicated_exc_body)

                if existing_state == OperationState.SENT:
                    self.logger.info(
                        f"Reusing pending operation for retry | TaskUUID: {task_uuid} | "
                        f"Previous state: {existing_state.value} | "
                        f"Partial results: {len(existing_op.get('results', []))}"
                    )
                    loop = asyncio.get_running_loop()
                    new_future = loop.create_future()
                    existing_op["future"] = new_future
                    existing_op["state"] = OperationState.SENT
                    if on_partial is not None:
                        existing_op["on_partial"] = on_partial
                    if complete_predicate is not None:
                        existing_op["complete_predicate"] = complete_predicate
                    if result_filter is not None:
                        existing_op["result_filter"] = result_filter
                    return new_future, False

                if existing_state == OperationState.DISCONNECTED:
                    self.logger.info(
                        f"Resending after disconnect | TaskUUID: {task_uuid} | "
                        f"Clearing {len(existing_op.get('results', []))} partial results"
                    )
                    loop = asyncio.get_running_loop()
                    new_future = loop.create_future()
                    existing_op["future"] = new_future
                    existing_op["results"] = []
                    existing_op["state"] = OperationState.SENT
                    if on_partial is not None:
                        existing_op["on_partial"] = on_partial
                    if complete_predicate is not None:
                        existing_op["complete_predicate"] = complete_predicate
                    if result_filter is not None:
                        existing_op["result_filter"] = result_filter
                    return new_future, True

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._pending_operations[task_uuid] = {
                "future": future,
                "expected": expected_results,
                "results": [],
                "state": OperationState.REGISTERED,
                "on_partial": on_partial,
                "complete_predicate": complete_predicate,
                "result_filter": result_filter
            }
            return future, True

    async def _mark_operation_sent(self, task_uuid: str) -> None:
        async with self._operations_lock:
            op = self._pending_operations.get(task_uuid)
            if op and op.get("state") == OperationState.REGISTERED:
                op["state"] = OperationState.SENT

    async def _unregister_pending_operation(self, task_uuid: str, force: bool = False) -> "Optional[List[Dict[str, Any]]]":
        async with self._operations_lock:
            op = self._pending_operations.get(task_uuid)
            if not op:
                return None

            if not force and op.get("state") == OperationState.DISCONNECTED:
                return op.get("results", [])

            return self._pending_operations.pop(task_uuid, {}).get("results")

    async def _handle_pending_operation_message(self, item: "Dict[str, Any]") -> bool:
        task_uuid = item.get("taskUUID")
        if not task_uuid:
            return False

        on_partial_callback = None
        async with self._operations_lock:
            op = self._pending_operations.get(task_uuid)
            if op is None:
                return False

            future = op["future"]

            if future.done():
                return True

            if self._is_error_response(item):
                if not future.done():
                    future.set_exception(RunwareAPIError(item))
                return True

            result_filter = op.get("result_filter")
            if result_filter is not None:
                if not result_filter(item):
                    return True

            op["results"].append(item)

            if op["on_partial"]:
                on_partial_callback = op["on_partial"]

            if op["complete_predicate"]:
                is_complete = op["complete_predicate"](item)
            else:
                is_complete = len(op["results"]) >= op["expected"]

            if is_complete and not future.done():
                logger.debug(f"Completing pending operation: {task_uuid}, results: {len(op['results'])}")
                future.set_result(op["results"])

        if on_partial_callback:
            try:
                if item.get("imageUUID"):
                    partial_images = [createImageFromResponse(item)]
                    on_partial_callback(partial_images, None)
                elif item.get("videoUUID") or item.get("mediaUUID"):
                    on_partial_callback([item], None)
                elif item.get("audioUUID"):
                    on_partial_callback([item], None)
            except Exception as e:
                logger.error(f"Error in on_partial callback: {e}")

        return True

    async def _handle_pending_operation_error(self, error: "Dict[str, Any]") -> bool:
        task_uuid = error.get("taskUUID")
        if not task_uuid:
            return False

        on_partial_callback = None
        error_obj = None
        async with self._operations_lock:
            op = self._pending_operations.get(task_uuid)
            if op is None:
                return False

            future = op["future"]

            if future.done():
                return True

            if op["on_partial"]:
                on_partial_callback = op["on_partial"]
                error_obj = IError(
                    error=True,
                    error_message=error.get("message", "Unknown error"),
                    task_uuid=task_uuid,
                    error_code=error.get("code"),
                    error_type=error.get("type"),
                    parameter=error.get("parameter"),
                    documentation=error.get("documentation"),
                )

            if not future.done():
                future.set_exception(RunwareAPIError(error))

        if on_partial_callback and error_obj is not None:
            try:
                on_partial_callback([], error_obj)
            except Exception as e:
                logger.error(f"Error in on_partial error callback: {e}")

        return True

    async def _do_reconnect(self, last_error: Optional[Exception] = None) -> bool:
        async with self._reconnect_lock:
            if self.connected() and self._connectionSessionUUID is not None:
                self.logger.info("Connection already re-established by another task")
                return True

            should_open_circuit = self._reconnection_manager.on_auth_failure()

            if should_open_circuit:
                self.logger.error("Authentication circuit breaker opened due to repeated failures")
                raise ConnectionError(
                    f"Authentication circuit breaker opened due to repeated failures. "
                    f"Last error: {str(last_error)}"
                )

            try:
                self.logger.info(f"Reconnecting after error: {str(last_error)}")

                self._invalidAPIkey = None
                self._connectionSessionUUID = None

                await self.connect()

                if not self.connected():
                    raise ConnectionError("Reconnection failed; WebSocket is not open")

                self.logger.info("Reconnection successful")
                return True

            except Exception as reconnect_error:
                self.logger.error(f"Error while reconnecting: {reconnect_error}", exc_info=True)
                return False

    async def _retry_with_reconnect(self, func, request_model, **func_kwargs):
        task_uuid = getattr(request_model, 'taskUUID', None)
        last_error = None

        try:
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    result = await func(request_model, **func_kwargs)
                    self._reconnection_manager.on_connection_success()
                    return result

                except Exception as e:
                    last_error = e

                    if isinstance(e, RunwareAPIError) and e.code == "conflictTaskUUID":
                        if attempt == 0:
                            raise
                        conflict_task_uuid = e.error_data.get("taskUUID") or getattr(request_model, 'taskUUID', None)
                        if conflict_task_uuid:
                            number_results = getattr(request_model, 'numberResults', 1) or 1
                            return await self._pollResults(
                                task_uuid=conflict_task_uuid,
                                number_results=number_results
                            )
                        raise

                    if not isinstance(e, ConnectionError):
                        raise

                    if attempt >= MAX_RETRY_ATTEMPTS - 1:
                        self.logger.error(f"Max retry attempts ({MAX_RETRY_ATTEMPTS}) exceeded")
                        raise ConnectionError(
                            f"Failed after {MAX_RETRY_ATTEMPTS} attempts. "
                            f"Last error: {last_error}"
                        )

                    reconnected = await self._do_reconnect(last_error)

                    if reconnected:
                        jitter = uniform(0.1, 0.5) * (attempt + 1)
                        await asyncio.sleep(jitter)
                    else:
                        delay = self._reconnection_manager.calculate_delay()
                        await asyncio.sleep(delay)
        finally:
            if task_uuid:
                await self._unregister_pending_operation(task_uuid, force=True)

    async def _retry_async_with_reconnect(self, func, request_model, task_type: str):
        task_uuid = getattr(request_model, 'taskUUID', None)
        delivery_method = getattr(request_model, 'deliveryMethod', None)

        try:
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    result = await func(request_model)
                    self._reconnection_manager.on_connection_success()
                    return result

                except Exception as e:
                    last_error = e

                    if isinstance(e, RunwareAPIError) and e.code == "conflictTaskUUID":
                        if attempt == 0:
                            raise

                        delivery_method_enum = None
                        if delivery_method is not None:
                            delivery_method_enum = (
                                EDeliveryMethod(delivery_method)
                                if isinstance(delivery_method, str)
                                else delivery_method
                            )

                        if task_type and task_uuid and delivery_method_enum is EDeliveryMethod.ASYNC:
                            return createAsyncTaskResponse({
                                "taskType": task_type,
                                "taskUUID": task_uuid
                            })

                        conflict_task_uuid = e.error_data.get("taskUUID") or task_uuid
                        if conflict_task_uuid:
                            number_results = getattr(request_model, 'numberResults', 1) or 1
                            return await self._pollResults(
                                task_uuid=conflict_task_uuid,
                                number_results=number_results
                            )

                        raise RunwareAPIError({
                            "code": "conflictTaskUUIDDuringRetries",
                            "message": "Lost connection during request submission",
                            "taskUUID": task_uuid
                        })

                    if not isinstance(e, ConnectionError):
                        raise

                    if attempt >= MAX_RETRY_ATTEMPTS - 1:
                        self.logger.error(f"Max retry attempts ({MAX_RETRY_ATTEMPTS}) exceeded")
                        raise ConnectionError(
                            f"Failed after {MAX_RETRY_ATTEMPTS} attempts. "
                            f"Last error: {last_error}"
                        )

                    reconnected = await self._do_reconnect(last_error)

                    if reconnected:
                        jitter = uniform(0.1, 0.5) * (attempt + 1)
                        await asyncio.sleep(jitter)
                    else:
                        delay = self._reconnection_manager.calculate_delay()
                        await asyncio.sleep(delay)
        finally:
            if task_uuid:
                await self._unregister_pending_operation(task_uuid, force=True)

    def _handle_error_response(self, response: Dict[str, Any]) -> None:
        """
        Handle error responses from the server.
        Raises ConnectionError for authentication errors, RunwareAPIError for others.
        """
        if not self._is_error_response(response):
            return

        # If an authentication error, raise ConnectionError to trigger retry
        if response.get("taskType") == "authentication" or response.get("code") == "missingApiKey":
            error_message = response.get("message", "Authentication error")
            self.logger.warning(f"Authentication error detected: {error_message}")
            raise ConnectionError(error_message)

        # For all other errors
        raise RunwareAPIError(response)

    def _create_safe_async_listener(self, async_func):
        def wrapper(m):
            task = asyncio.create_task(async_func(m))
            self._listener_tasks.add(task)

            def handle_task_exception(t):
                self._listener_tasks.discard(t)
                if not t.cancelled():
                    try:
                        t.result()
                    except Exception as e:
                        logger.error(f"Unhandled exception in async listener: {e}", exc_info=True)

            task.add_done_callback(handle_task_exception)
            return None

        return wrapper

    async def _cleanup_listener_tasks(self):
        if not self._listener_tasks:
            return

        self.logger.info(f"Cleaning up {len(self._listener_tasks)} listener tasks")

        for task in list(self._listener_tasks):
            if not task.done():
                task.cancel()

        if self._listener_tasks:
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)

        self._listener_tasks.clear()
        self.logger.info("All listener tasks cleaned up")

    def isWebsocketReadyState(self) -> bool:
        if self._ws is None:
            return False
        return self._ws.state is State.OPEN

    def isAuthenticated(self):
        return self._connectionSessionUUID is not None

    def addListener(
        self,
        lis: Callable[[Any], Any],
        check: Callable[[Any], Any],
        groupKey: Optional[str] = None,
    ) -> Dict[str, Callable[[], None]]:
        # Get the current frame
        current_frame = inspect.currentframe()

        # Get the caller's frame
        caller_frame = current_frame.f_back

        # Get the caller's function name
        caller_name = caller_frame.f_code.co_name

        # Get the caller's line number
        caller_line_number = caller_frame.f_lineno

        debug_message = f"Listener {self.addListener.__name__} created by {caller_name} at line {caller_line_number} with listener: {lis} and check: {check}"
        # logger.debug(debug_message)

        if not lis or not check:
            raise ValueError("Listener and check functions are required")

        def listener(msg: Any) -> None:
            if not lis or not check:
                raise ValueError("Listener and check functions are required")
            if msg.get("error"):
                lis(msg)
            elif check(msg):
                lis(msg)

        groupListener: ListenerType = ListenerType(
            key=getUUID(),
            listener=listener,
            group_key=groupKey,
            debug_message=debug_message,
        )
        self._listeners.append(groupListener)

        def destroy() -> None:
            self._listeners = removeListener(self._listeners, groupListener)

        return {"destroy": destroy}

    def handle_connection_response(self, m):
        if m.get("error"):
            if m["errorId"] == 19:
                self._invalidAPIkey = "Invalid API key"
            else:
                self._invalidAPIkey = "Error connection"
            return

        self._connectionSessionUUID = m.get("newConnectionSessionUUID", {}).get(
            "connectionSessionUUID"
        )
        self._invalidAPIkey = None

    async def photoMaker(self, requestPhotoMaker: "IPhotoMaker") -> "Union[List[IImage], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._photoMaker, requestPhotoMaker)

    async def _photoMaker(self, requestPhotoMaker: "IPhotoMaker") -> "Union[List[IImage], IAsyncTaskResponse]":
        await self.ensureConnection()

        task_uuid = requestPhotoMaker.taskUUID or getUUID()
        requestPhotoMaker.taskUUID = task_uuid

        for i, image in enumerate(requestPhotoMaker.inputImages):
            if isLocalFile(image) and not str(image).startswith("http"):
                requestPhotoMaker.inputImages[i] = await fileToBase64(image)

        prompt = f"{requestPhotoMaker.positivePrompt}".strip()
        request_object = {
            "taskUUID": requestPhotoMaker.taskUUID,
            "model": requestPhotoMaker.model,
            "positivePrompt": prompt,
            "numberResults": requestPhotoMaker.numberResults,
            "height": requestPhotoMaker.height,
            "width": requestPhotoMaker.width,
            "taskType": ETaskType.PHOTO_MAKER.value,
            "style": requestPhotoMaker.style,
            "strength": requestPhotoMaker.strength,
            **(
                {"inputImages": requestPhotoMaker.inputImages}
                if requestPhotoMaker.inputImages
                else {}
            ),
            **(
                {"steps": requestPhotoMaker.steps}
                if requestPhotoMaker.steps
                else {}
            ),
        }

        if requestPhotoMaker.outputFormat is not None:
            request_object["outputFormat"] = requestPhotoMaker.outputFormat
        if requestPhotoMaker.includeCost:
            request_object["includeCost"] = requestPhotoMaker.includeCost
        if requestPhotoMaker.outputType:
            request_object["outputType"] = requestPhotoMaker.outputType
        if requestPhotoMaker.webhookURL:
            request_object["webhookURL"] = requestPhotoMaker.webhookURL
            return await self._handleWebhookRequest(
                request_object=request_object,
                task_uuid=task_uuid,
                task_type="photoMaker",
                debug_key="photo-maker-webhook"
            )

        numberOfResults = requestPhotoMaker.numberResults

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=numberOfResults,
            complete_predicate=None,
            result_filter=lambda r: r.get("imageUUID") is not None
        )

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=IMAGE_INFERENCE_TIMEOUT / 1000)

            unique_results = {}
            for made_photo in results:
                image_uuid = made_photo.get("imageUUID")
                if image_uuid and image_uuid not in unique_results:
                    unique_results[image_uuid] = made_photo

            if not unique_results:
                raise Exception(f"No valid photoMaker results received | TaskUUID: {task_uuid}")

            return instantiateDataclassList(IImage, list(unique_results.values()))

        except asyncio.TimeoutError:
            op = self._pending_operations.get(task_uuid)
            partial_count = len(op["results"]) if op else 0
            raise Exception(
                f"Timeout waiting for photoMaker | TaskUUID: {task_uuid} | "
                f"Expected: {numberOfResults} | Received: {partial_count} | "
                f"Timeout: {IMAGE_INFERENCE_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def imageInference(
            self, requestImage: "IImageInference"
    ) -> "Union[List[IImage], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._imageInference,
                requestImage,
                task_type=ETaskType.IMAGE_INFERENCE.value
            )

    async def _imageInference(
            self, requestImage: "IImageInference"
    ) -> "Union[List[IImage], IAsyncTaskResponse]":
        await self.ensureConnection()

        control_net_data: "List[IControlNet]" = []
        requestImage.taskUUID = requestImage.taskUUID or getUUID()
        requestImage.maskImage = await process_image(requestImage.maskImage)
        requestImage.seedImage = await process_image(requestImage.seedImage)

        if requestImage.referenceImages:
            requestImage.referenceImages = await process_image(requestImage.referenceImages)

        if requestImage.controlNet:
            for control_data in requestImage.controlNet:
                image_uploaded = await self.uploadImage(control_data.guideImage)
                if not image_uploaded:
                    return []
                if hasattr(control_data, "preprocessor"):
                    control_data.preprocessor = control_data.preprocessor.value
                control_data.guideImage = image_uploaded.imageUUID
                control_net_data.append(control_data)

        prompt = requestImage.positivePrompt.strip() if requestImage.positivePrompt else None
        control_net_data_dicts = [asdict(item) for item in control_net_data]

        instant_id_data = {}
        if requestImage.instantID:
            instant_id_data = {
                k: v
                for k, v in vars(requestImage.instantID).items()
                if v is not None
            }
            if "inputImage" in instant_id_data:
                instant_id_data["inputImage"] = await process_image(instant_id_data["inputImage"])
            if "poseImage" in instant_id_data:
                instant_id_data["poseImage"] = await process_image(instant_id_data["poseImage"])

        ip_adapters_data = []
        if requestImage.ipAdapters:
            for ip_adapter in requestImage.ipAdapters:
                ip_adapter_data = {
                    k: v for k, v in vars(ip_adapter).items() if v is not None
                }
                if "guideImage" in ip_adapter_data:
                    ip_adapter_data["guideImage"] = await process_image(ip_adapter_data["guideImage"])
                ip_adapters_data.append(ip_adapter_data)

        ace_plus_plus_data = {}
        if requestImage.acePlusPlus:
            ace_plus_plus_data = {
                "inputImages": [],
                "repaintingScale": requestImage.acePlusPlus.repaintingScale,
                "type": requestImage.acePlusPlus.taskType,
            }
            if requestImage.acePlusPlus.inputImages:
                ace_plus_plus_data["inputImages"] = await process_image(requestImage.acePlusPlus.inputImages)
            if requestImage.acePlusPlus.inputMasks:
                ace_plus_plus_data["inputMasks"] = await process_image(requestImage.acePlusPlus.inputMasks)

        pulid_data = {}
        if requestImage.puLID:
            pulid_data = {
                "inputImages": [],
                "idWeight": requestImage.puLID.idWeight,
                "trueCFGScale": requestImage.puLID.trueCFGScale,
                "CFGStartStep": requestImage.puLID.CFGStartStep,
                "CFGStartStepPercentage": requestImage.puLID.CFGStartStepPercentage,
            }
            if requestImage.puLID.inputImages:
                pulid_data["inputImages"] = await process_image(requestImage.puLID.inputImages)

        request_object = self._buildImageRequest(
            requestImage, prompt, control_net_data_dicts,
            instant_id_data, ip_adapters_data, ace_plus_plus_data, pulid_data
        )

        delivery_method_enum = EDeliveryMethod(requestImage.deliveryMethod) if isinstance(requestImage.deliveryMethod,
                                                                                          str) else requestImage.deliveryMethod
        task_uuid = requestImage.taskUUID
        number_results = requestImage.numberResults or 1

        if delivery_method_enum is EDeliveryMethod.ASYNC:
            if requestImage.webhookURL:
                request_object["webhookURL"] = requestImage.webhookURL
                return await self._handleWebhookRequest(
                    request_object=request_object,
                    task_uuid=task_uuid,
                    task_type="imageInference",
                    debug_key="image-inference-webhook"
                )

            future, should_send = await self._register_pending_operation(
                task_uuid,
                expected_results=1,
                complete_predicate=lambda r: True
            )

            try:
                if should_send:
                    await self.send([request_object])
                    await self._mark_operation_sent(task_uuid)
                results = await asyncio.wait_for(future, timeout=IMAGE_INITIAL_TIMEOUT / 1000)
                response = results[0]
                self._handle_error_response(response)

                if response.get("status") == "success" or response.get("imageUUID") is not None:
                    return instantiateDataclassList(IImage, results)

                return createAsyncTaskResponse(response)
            except asyncio.TimeoutError:
                raise ConnectionError(
                    f"Timeout waiting for async acknowledgment | TaskUUID: {task_uuid} | "
                    f"Timeout: {IMAGE_INITIAL_TIMEOUT}ms"
                )
            finally:
                await self._unregister_pending_operation(task_uuid)

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=number_results,
            on_partial=requestImage.onPartialImages,
            complete_predicate=None,
            result_filter=lambda r: r.get("imageUUID") is not None
        )

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=IMAGE_INFERENCE_TIMEOUT / 1000)

            if not results:
                raise Exception(f"No results received | TaskUUID: {task_uuid}")

            return instantiateDataclassList(IImage, results)

        except asyncio.TimeoutError:
            op = self._pending_operations.get(task_uuid)
            partial_count = len(op["results"]) if op else 0

            if op and op["results"]:
                self.logger.warning(
                    f"Timeout but returning {partial_count} partial results | "
                    f"TaskUUID: {task_uuid} | Expected: {number_results}"
                )
                return instantiateDataclassList(IImage, op["results"])

            raise Exception(
                f"Timeout waiting for image inference | TaskUUID: {task_uuid} | "
                f"Expected: {number_results} | Received: {partial_count} | "
                f"Timeout: {IMAGE_INFERENCE_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def imageCaption(self, requestImageToText: "IImageCaption") -> "Union[IImageToText, IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._imageCaption, requestImageToText)

    async def _imageCaption(self, requestImageToText: IImageCaption) -> Union[IImageToText, IAsyncTaskResponse]:
        await self.ensureConnection()
        return await self._requestImageToText(requestImageToText)

    async def _requestImageToText(
        self, requestImageToText: "IImageCaption"
    ) -> "Union[IImageToText, IAsyncTaskResponse]":
        # Prepare image list - inputImages is primary, inputImage is convenience
        if requestImageToText.inputImages is not None:
            images_to_process = requestImageToText.inputImages
        elif requestImageToText.inputImage is not None:
            # Single image provided via inputImage - convert to array
            images_to_process = [requestImageToText.inputImage]
        else:
            raise ValueError("Either inputImages or inputImage must be provided")

        # Set inputImage to inputImages[0] if not already provided
        actual_input_image = requestImageToText.inputImage
        if actual_input_image is None and images_to_process:
            actual_input_image = images_to_process[0]
        # Upload all images
        uploaded_images = []
        for image in images_to_process:
            image_uploaded = await self.uploadImage(image)
            if not image_uploaded or not image_uploaded.imageUUID:
                return None
            uploaded_images.append(image_uploaded.imageUUID)

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_CAPTION.value,
            "taskUUID": taskUUID,
        }
        # Add either inputImage or inputImages, but not both (API requirement)
        if len(uploaded_images) == 1:
            # Single image - use inputImage parameter
            task_params["inputImage"] = uploaded_images[0]
        else:
            # Multiple images - use inputImages parameter
            task_params["inputImages"] = uploaded_images

        # Add model parameter only if specified - backend handles default
        if requestImageToText.model is not None:
            task_params["model"] = requestImageToText.model

        # Add template parameter if specified
        if requestImageToText.template is not None:
            task_params["template"] = requestImageToText.template
            # When using template, do NOT include prompt parameter
        else:
            # Use the provided prompt when no template
            task_params["prompt"] = requestImageToText.prompt

        # Add optional parameters if they are provided
        if requestImageToText.includeCost:
            task_params["includeCost"] = requestImageToText.includeCost
        if requestImageToText.webhookURL:
            task_params["webhookURL"] = requestImageToText.webhookURL
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="imageCaption",
                debug_key="image-caption-webhook"
            )

        future, should_send = await self._register_pending_operation(
            taskUUID,
            expected_results=1,
            complete_predicate=lambda r: True
        )

        try:
            if should_send:
                await self.send([task_params])
                await self._mark_operation_sent(taskUUID)
            results = await asyncio.wait_for(future, timeout=IMAGE_OPERATION_TIMEOUT / 1000)
            response = results[0]
            self._handle_error_response(response)
            return createImageToTextFromResponse(response)
        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for image caption | TaskUUID: {taskUUID} | "
                f"Timeout: {IMAGE_OPERATION_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(taskUUID)

    async def videoCaption(self, requestVideoCaption: "IVideoCaption") -> "Union[List[IVideoToText], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._requestVideoCaption,
                requestVideoCaption,
                task_type=ETaskType.VIDEO_CAPTION.value
            )

    async def _requestVideoCaption(
            self, requestVideoCaption: "IVideoCaption"
    ) -> "Union[List[IVideoToText], IAsyncTaskResponse]":
        await self.ensureConnection()
        taskUUID = requestVideoCaption.taskUUID or getUUID()

        # Create the request object
        task_params = {
            "taskType": ETaskType.VIDEO_CAPTION.value,
            "taskUUID": taskUUID,
            "model": requestVideoCaption.model,
            "inputs": {
                "video": requestVideoCaption.inputs.video
            },
            "deliveryMethod": requestVideoCaption.deliveryMethod,
        }

        # Add optional parameters
        if requestVideoCaption.includeCost is not None:
            task_params["includeCost"] = requestVideoCaption.includeCost
        if requestVideoCaption.webhookURL:
            task_params["webhookURL"] = requestVideoCaption.webhookURL
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="caption",
                debug_key="video-caption-webhook"
            )

        await self.send([task_params])

        return await self._pollResults(
            task_uuid=taskUUID,
            number_results=1,
        )

    async def videoBackgroundRemoval(self,
                                     requestVideoBackgroundRemoval: "IVideoBackgroundRemoval") -> "Union[List[IVideo], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._requestVideoBackgroundRemoval,
                requestVideoBackgroundRemoval,
                task_type=ETaskType.VIDEO_BACKGROUND_REMOVAL.value
            )

    async def _requestVideoBackgroundRemoval(
            self, requestVideoBackgroundRemoval: "IVideoBackgroundRemoval"
    ) -> "Union[List[IVideo], IAsyncTaskResponse]":
        await self.ensureConnection()
        taskUUID = requestVideoBackgroundRemoval.taskUUID or getUUID()

        # Create the request object
        task_params = {
            "taskType": ETaskType.VIDEO_BACKGROUND_REMOVAL.value,  # "removeBackground"
            "taskUUID": taskUUID,
            "model": requestVideoBackgroundRemoval.model,
            "inputs": {
                "video": requestVideoBackgroundRemoval.inputs.video
            },
            "deliveryMethod": requestVideoBackgroundRemoval.deliveryMethod,
        }

        # Add optional parameters
        if requestVideoBackgroundRemoval.outputFormat:
            task_params["outputFormat"] = requestVideoBackgroundRemoval.outputFormat
        if requestVideoBackgroundRemoval.includeCost is not None:
            task_params["includeCost"] = requestVideoBackgroundRemoval.includeCost
        if requestVideoBackgroundRemoval.webhookURL:
            task_params["webhookURL"] = requestVideoBackgroundRemoval.webhookURL
        if requestVideoBackgroundRemoval.settings:
            # Convert IBackgroundRemovalSettings to dict, filtering out None values
            settings_dict = {
                k: v
                for k, v in vars(requestVideoBackgroundRemoval.settings).items()
                if v is not None
            }
            task_params["settings"] = settings_dict

        if requestVideoBackgroundRemoval.webhookURL:
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="removeBackground",
                debug_key="video-background-removal-webhook"
            )

        return await self._handleInitialVideoResponse(
            request_object=task_params,
            task_uuid=taskUUID,
            number_results=1,
            delivery_method=requestVideoBackgroundRemoval.deliveryMethod,
            webhook_url=None,
            debug_key="video-background-removal-initial"
        )

    async def videoUpscale(self, requestVideoUpscale: "IVideoUpscale") -> "Union[List[IVideo], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._requestVideoUpscale,
                requestVideoUpscale,
                task_type=ETaskType.VIDEO_UPSCALE.value
            )

    async def _requestVideoUpscale(self, requestVideoUpscale: "IVideoUpscale") -> "Union[List[IVideo], IAsyncTaskResponse]":
        await self.ensureConnection()
        taskUUID = requestVideoUpscale.taskUUID or getUUID()

        # Create the request object
        task_params = {
            "taskType": ETaskType.VIDEO_UPSCALE.value,  # "upscale"
            "taskUUID": taskUUID,
            "model": requestVideoUpscale.model,
            "inputs": {
                "video": requestVideoUpscale.inputs.video
            },
            "deliveryMethod": requestVideoUpscale.deliveryMethod,
        }

        # Add optional parameters
        if requestVideoUpscale.upscaleFactor is not None:
            task_params["upscaleFactor"] = requestVideoUpscale.upscaleFactor
        if requestVideoUpscale.outputFormat:
            task_params["outputFormat"] = requestVideoUpscale.outputFormat
        if requestVideoUpscale.outputType:
            task_params["outputType"] = requestVideoUpscale.outputType
        if requestVideoUpscale.includeCost is not None:
            task_params["includeCost"] = requestVideoUpscale.includeCost
        if requestVideoUpscale.webhookURL:
            task_params["webhookURL"] = requestVideoUpscale.webhookURL

        if requestVideoUpscale.webhookURL:
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="upscale",
                debug_key="video-upscale-webhook"
            )

        return await self._handleInitialVideoResponse(
            request_object=task_params,
            task_uuid=taskUUID,
            number_results=1,
            delivery_method=requestVideoUpscale.deliveryMethod,
            webhook_url=None,
            debug_key="video-upscale-initial"
        )

    async def imageBackgroundRemoval(
        self, removeImageBackgroundPayload: "IImageBackgroundRemoval"
    ) -> "Union[List[IImage], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._removeImageBackground, removeImageBackgroundPayload)

    async def _removeImageBackground(
        self, removeImageBackgroundPayload: "IImageBackgroundRemoval"
    ) -> "Union[List[IImage], IAsyncTaskResponse]":
        await self.ensureConnection()
        inputImage = removeImageBackgroundPayload.inputImage

        image_uploaded = await self.uploadImage(inputImage)

        if not image_uploaded or not image_uploaded.imageUUID:
            return []

        if removeImageBackgroundPayload.taskUUID is not None:
            taskUUID = removeImageBackgroundPayload.taskUUID
        else:
            taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_BACKGROUND_REMOVAL.value,
            "taskUUID": taskUUID,
            "inputImage": image_uploaded.imageUUID,
        }

        # Add optional parameters if they are provided
        if removeImageBackgroundPayload.outputType is not None:
            task_params["outputType"] = removeImageBackgroundPayload.outputType
        if removeImageBackgroundPayload.outputFormat is not None:
            task_params["outputFormat"] = removeImageBackgroundPayload.outputFormat
        if removeImageBackgroundPayload.includeCost:
            task_params["includeCost"] = removeImageBackgroundPayload.includeCost
        if removeImageBackgroundPayload.model:
            task_params["model"] = removeImageBackgroundPayload.model
        if removeImageBackgroundPayload.outputQuality:
            task_params["outputQuality"] = removeImageBackgroundPayload.outputQuality
        if removeImageBackgroundPayload.webhookURL:
            task_params["webhookURL"] = removeImageBackgroundPayload.webhookURL

        # Handle settings if provided - convert dataclass to dictionary and add non-None values
        if removeImageBackgroundPayload.settings:
            settings_dict = {
                k: v
                for k, v in vars(removeImageBackgroundPayload.settings).items()
                if v is not None
            }
            task_params.update(settings_dict)

        # Add provider settings if provided
        if removeImageBackgroundPayload.providerSettings:
            self._addImageProviderSettings(task_params, removeImageBackgroundPayload)

        # Add safety settings if provided
        if removeImageBackgroundPayload.safety:
            self._addSafetySettings(task_params, removeImageBackgroundPayload.safety)

        if removeImageBackgroundPayload.webhookURL:
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="imageBackgroundRemoval",
                debug_key="image-background-removal-webhook"
            )

        future, should_send = await self._register_pending_operation(
            taskUUID,
            expected_results=1,
            complete_predicate=None,
            result_filter=lambda r: r.get("imageUUID") is not None
        )

        try:
            if should_send:
                await self.send([task_params])
                await self._mark_operation_sent(taskUUID)
            results = await asyncio.wait_for(future, timeout=IMAGE_OPERATION_TIMEOUT / 1000)
            response = results[0]
            self._handle_error_response(response)
            image = createImageFromResponse(response)
            return [image]
        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for background removal | TaskUUID: {taskUUID} | "
                f"Timeout: {IMAGE_OPERATION_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(taskUUID)

    async def imageUpscale(self, upscaleGanPayload: "IImageUpscale") -> "Union[List[IImage], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._imageUpscale, upscaleGanPayload)

    async def _imageUpscale(self, upscaleGanPayload: IImageUpscale) -> Union[List[IImage], IAsyncTaskResponse]:
        await self.ensureConnection()
        return await self._upscaleGan(upscaleGanPayload)

    async def _upscaleGan(self, upscaleGanPayload: "IImageUpscale") -> "Union[List[IImage], IAsyncTaskResponse]":
        # Support both inputImage (legacy) and inputs.image (new format)
        inputImage = upscaleGanPayload.inputImage
        if not inputImage and upscaleGanPayload.inputs and upscaleGanPayload.inputs.image:
            inputImage = upscaleGanPayload.inputs.image

        if not inputImage:
            raise ValueError("Either inputImage or inputs.image must be provided")

        image_uploaded = await self.uploadImage(inputImage)

        if not image_uploaded or not image_uploaded.imageUUID:
            return []

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_UPSCALE.value,
            "taskUUID": taskUUID,
            "upscaleFactor": upscaleGanPayload.upscaleFactor,
        }

        # Use inputs.image format if inputs is provided, otherwise use inputImage (legacy)
        if upscaleGanPayload.inputs and upscaleGanPayload.inputs.image:
            task_params["inputs"] = {"image": image_uploaded.imageUUID}
        else:
            task_params["inputImage"] = image_uploaded.imageUUID

        # Add model parameter if specified
        if upscaleGanPayload.model is not None:
            task_params["model"] = upscaleGanPayload.model

        # Add settings if provided
        if upscaleGanPayload.settings is not None:
            settings_dict = asdict(upscaleGanPayload.settings)
            # Remove None values
            settings_dict = {k: v for k, v in settings_dict.items() if v is not None}
            if settings_dict:
                task_params["settings"] = settings_dict

        # Add optional parameters if they are provided
        if upscaleGanPayload.outputType is not None:
            task_params["outputType"] = upscaleGanPayload.outputType
        if upscaleGanPayload.outputFormat is not None:
            task_params["outputFormat"] = upscaleGanPayload.outputFormat
        if upscaleGanPayload.includeCost:
            task_params["includeCost"] = upscaleGanPayload.includeCost
        if upscaleGanPayload.webhookURL:
            task_params["webhookURL"] = upscaleGanPayload.webhookURL

        # Add provider settings if provided
        if upscaleGanPayload.providerSettings:
            self._addImageProviderSettings(task_params, upscaleGanPayload)

        # Add safety settings if provided
        if upscaleGanPayload.safety:
            self._addSafetySettings(task_params, upscaleGanPayload.safety)

        if upscaleGanPayload.webhookURL:
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="imageUpscale",
                debug_key="image-upscale-webhook"
            )

        future, should_send = await self._register_pending_operation(
            taskUUID,
            expected_results=1,
            complete_predicate=None,
            result_filter=lambda r: r.get("imageUUID") is not None
        )

        try:
            if should_send:
                await self.send([task_params])
                await self._mark_operation_sent(taskUUID)
            results = await asyncio.wait_for(future, timeout=IMAGE_OPERATION_TIMEOUT / 1000)
            response = results[0]
            self._handle_error_response(response)
            image = createImageFromResponse(response)
            return [image]
        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for image upscale | TaskUUID: {taskUUID} | "
                f"Timeout: {IMAGE_OPERATION_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(taskUUID)

    async def imageVectorize(self, vectorizePayload: "IVectorize") -> "Union[List[IImage], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._vectorize, vectorizePayload)

    async def _vectorize(self, vectorizePayload: "IVectorize") -> Union[List["IImage"], "IAsyncTaskResponse"]:
        await self.ensureConnection()
        # Process the image from inputs
        input_image = vectorizePayload.inputs.image

        if not input_image:
            raise ValueError("Image is required in inputs for vectorize task")

        # Upload the image if it's a local file
        image_uploaded = await self.uploadImage(input_image)

        if not image_uploaded or not image_uploaded.imageUUID:
            return []

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_VECTORIZE.value,
            "taskUUID": taskUUID,
            "inputs": {
                "image": image_uploaded.imageUUID
            }
        }

        # Add optional parameters if they are provided
        if vectorizePayload.model is not None:
            task_params["model"] = vectorizePayload.model
        if vectorizePayload.outputType is not None:
            task_params["outputType"] = vectorizePayload.outputType
        if vectorizePayload.outputFormat is not None:
            task_params["outputFormat"] = vectorizePayload.outputFormat
        if vectorizePayload.includeCost:
            task_params["includeCost"] = vectorizePayload.includeCost
        if vectorizePayload.webhookURL:
            task_params["webhookURL"] = vectorizePayload.webhookURL
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="vectorize",
                debug_key="image-vectorize-webhook"
            )

        future, should_send = await self._register_pending_operation(
            taskUUID,
            expected_results=1,
            complete_predicate=None,
            result_filter=lambda r: r.get("imageUUID") is not None
        )

        try:

            if should_send:
                await self.send([task_params])
                await self._mark_operation_sent(taskUUID)
            results = await asyncio.wait_for(future, timeout=IMAGE_OPERATION_TIMEOUT / 1000)

            if not results:
                raise Exception(f"No results received | TaskUUID: {taskUUID}")

            for result in results:
                if "code" in result or "errors" in result:
                    raise RunwareAPIError(result)

            return instantiateDataclassList(IImage, results)

        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for vectorize | TaskUUID: {taskUUID} | "
                f"Timeout: {IMAGE_OPERATION_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(taskUUID)

    async def promptEnhance(
        self, promptEnhancer: "IPromptEnhance"
    ) -> "Union[List[IEnhancedPrompt], IAsyncTaskResponse]":
        """
        Enhance the given prompt by generating multiple versions of it.

        :param promptEnhancer: An IPromptEnhancer object containing the prompt details.
        :return: A list of IEnhancedPrompt objects representing the enhanced versions of the prompt.
        :raises: Any error that occurs during the enhancement process.
        """
        async with self._request_semaphore:
            try:
                await self.ensureConnection()
                return await asyncRetry(lambda: self._enhancePrompt(promptEnhancer))
            except Exception as e:
                raise e

    async def _enhancePrompt(
        self, promptEnhancer: "IPromptEnhance"
    ) -> "Union[List[IEnhancedPrompt], IAsyncTaskResponse]":
        prompt = promptEnhancer.prompt
        promptMaxLength = getattr(promptEnhancer, "promptMaxLength", 380)

        promptVersions = promptEnhancer.promptVersions or 1

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.PROMPT_ENHANCE.value,
            "taskUUID": taskUUID,
            "prompt": prompt,
            "promptMaxLength": promptMaxLength,
            "promptVersions": promptVersions,
        }

        # Add optional parameters if they are provided
        if promptEnhancer.includeCost:
            task_params["includeCost"] = promptEnhancer.includeCost

        if promptEnhancer.webhookURL:
            task_params["webhookURL"] = promptEnhancer.webhookURL
            return await self._handleWebhookRequest(
                request_object=task_params,
                task_uuid=taskUUID,
                task_type="promptEnhance",
                debug_key="prompt-enhance-webhook"
            )

        future, should_send = await self._register_pending_operation(
            taskUUID,
            expected_results=promptVersions,
            complete_predicate=None,
            result_filter=lambda r: r.get("text") is not None
        )

        try:
            if should_send:
                await self.send([task_params])
                await self._mark_operation_sent(taskUUID)
            results = await asyncio.wait_for(future, timeout=PROMPT_ENHANCE_TIMEOUT / 1000)

            if not results:
                raise Exception(f"No results received | TaskUUID: {taskUUID}")

            for result in results:
                if "code" in result:
                    raise RunwareAPIError(result)

            # Transform the response to a list of IEnhancedPrompt objects
            enhanced_prompts = createEnhancedPromptsFromResponse(results)
            return list(set(enhanced_prompts))

        except asyncio.TimeoutError:
            op = self._pending_operations.get(taskUUID)
            partial_count = len(op["results"]) if op else 0
            raise Exception(
                f"Timeout waiting for prompt enhance | TaskUUID: {taskUUID} | "
                f"Expected: {promptVersions} | Received: {partial_count} | "
                f"Timeout: {PROMPT_ENHANCE_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(taskUUID)

    async def uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        request = IUploadImageRequest(file=file, taskUUID=getUUID())
        return await self._retry_with_reconnect(self._uploadImage, request)

    async def _uploadImage(self, request: IUploadImageRequest) -> "Optional[UploadImageType]":
        await self.ensureConnection()

        file = request.file
        task_uuid = request.taskUUID
        local_file = True

        if isinstance(file, str):
            if os.path.exists(file):
                local_file = True
            else:
                local_file = isLocalFile(file)

                # Check if it's a base64 string (with or without data URI prefix)
                if file.startswith("data:") or re.match(
                    r"^[A-Za-z0-9+/]+={0,2}$", file
                ):
                    # Assume it's a base64 string (with or without data URI prefix)
                    local_file = False

            if not local_file:
                return UploadImageType(
                    imageUUID=file,
                    imageURL=file,
                    taskUUID=task_uuid,
                )

        # Convert file to base64 (handles both File objects and string paths)
        file_data = await fileToBase64(file)

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=lambda r: True
        )

        try:
            if should_send:
                await self.send([
                    {
                        "taskType": ETaskType.IMAGE_UPLOAD.value,
                        "taskUUID": task_uuid,
                        "image": file_data,
                    }
                ])
                await self._mark_operation_sent(task_uuid)

            results = await asyncio.wait_for(future, timeout=IMAGE_UPLOAD_TIMEOUT / 1000)
            response = results[0]
            self._handle_error_response(response)

            return UploadImageType(
                imageUUID=response["imageUUID"],
                imageURL=response["imageURL"],
                taskUUID=response["taskUUID"],
            )
        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for image upload | TaskUUID: {task_uuid} | "
                f"Timeout: {IMAGE_UPLOAD_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def uploadMedia(self, media_url: str) -> "Optional[MediaStorageType]":
        request = IUploadMediaRequest(media_url=media_url, taskUUID=getUUID())
        return await self._retry_with_reconnect(self._uploadMedia, request)

    async def _uploadMedia(self, request: IUploadMediaRequest) -> "Optional[MediaStorageType]":
        await self.ensureConnection()

        media_url = request.media_url
        task_uuid = request.taskUUID
        media_data = media_url

        if isinstance(media_url, str):
            if os.path.exists(media_url):
                # Local file - convert to base64
                media_data = await fileToBase64(media_url)
                # Strip the data URI prefix for media storage API
                if media_data.startswith("data:"):
                    media_data = media_data.split(",", 1)[1]
            # For URLs and base64 strings, send them directly to the API

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=lambda r: r.get("mediaUUID") is not None
        )

        try:
            if should_send:
                await self.send(
                    [
                        {
                            "taskType": ETaskType.MEDIA_STORAGE.value,
                            "taskUUID": task_uuid,
                            "operation": "upload",
                            "media": media_data,
                        }
                    ]
                )
                await self._mark_operation_sent(task_uuid)

            results = await asyncio.wait_for(future, timeout=self._timeout / 1000)
            response = results[0]

            self._handle_error_response(response)

            if response:
                return MediaStorageType(
                    mediaUUID=response["mediaUUID"],
                    taskUUID=response["taskUUID"],
                )
            return None

        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for media upload | TaskUUID: {task_uuid} | "
                f"Timeout: {self._timeout}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def listenToImages(
        self,
        onPartialImages: Optional[Callable[[List[IImage], Optional[IError]], None]],
        taskUUID: str,
        groupKey: LISTEN_TO_IMAGES_KEY,
    ) -> Dict[str, Callable[[], None]]:
        logger.debug("Setting up images listener for taskUUID: %s", taskUUID)

        async def listen_to_images_lis(m: Dict[str, Any]) -> None:
            if isinstance(m.get("data"), list):
                images = [
                    img
                    for img in m["data"]
                    if img.get("taskType") in ["imageInference", "vectorize"]
                    and img.get("taskUUID") == taskUUID
                ]

                if images:
                    async with self._images_lock:
                        self._globalImages.extend(images)

                    try:
                        partial_images = instantiateDataclassList(IImage, images)
                        if onPartialImages:
                            onPartialImages(
                                partial_images, None
                            )
                    except Exception as e:
                        logger.error(
                            f"Error occurred in user on_partial_images callback function: {e}"
                        )
            elif isinstance(m.get("errors"), list):
                errors = [
                    error for error in m["errors"] if error.get("taskUUID") == taskUUID
                ]
                if errors:
                    error = IError(
                        error=True,
                        error_message=errors[0].get("message", "Unknown error"),
                        task_uuid=errors[0].get("taskUUID", ""),
                        error_code=errors[0].get("code"),
                        error_type=errors[0].get("type"),
                        parameter=errors[0].get("parameter"),
                        documentation=errors[0].get("documentation"),
                    )
                    self._globalError = error
                    if onPartialImages:
                        onPartialImages(
                            [], self._globalError
                        )

        def listen_to_images_check(m):
            logger.debug("Images check message: %s", m)
            image_inference_check = isinstance(m.get("data"), list) and any(
                item.get("taskType") in ["imageInference", "vectorize"] for item in m["data"]
            )
            error_check = isinstance(m.get("errors"), list) and any(
                error.get("taskUUID") == taskUUID for error in m["errors"]
            )
            error_code_check = (
                True
                if any([error.get("code") for error in m.get("errors", [])])
                else False
            )
            if error_code_check:
                self._globalError = IError(
                    error=True,
                    error_message=f"Error in image inference: {m.get('errors')}",
                    task_uuid=taskUUID,
                )

            response = image_inference_check or error_check
            return response

        temp_listener = self.addListener(
            check=listen_to_images_check,
            lis=self._create_safe_async_listener(listen_to_images_lis),
            groupKey=groupKey
        )

        logger.debug("listenToImages :: Temp listener: %s", temp_listener)

        return temp_listener

    def globalListener(self, taskUUID: str) -> Dict[str, Callable[[], None]]:
        """
        Set up a global listener to capture specific messages based on the provided taskUUID.

        :param taskUUID: The unique identifier of the task associated with the listener.
        :return: A dictionary containing a 'destroy' function to remove the listener.
        """
        logger.debug("Setting up global listener for taskUUID: %s", taskUUID)

        async def global_lis(m: Dict[str, Any]) -> None:
            logger.debug("Global listener message: %s", m)
            logger.debug("Global listener taskUUID: %s", taskUUID)

            async with self._messages_lock:
                if m.get("error"):
                    self._globalMessages[taskUUID] = m
                    return

                value = accessDeepObject(
                    taskUUID, m
                )

                if isinstance(value, list):
                    for v in value:
                        self._globalMessages[v["taskUUID"]] = self._globalMessages.get(
                            v["taskUUID"], []
                        ) + [v]
                        logger.debug("Global messages v: %s", v)
                        logger.debug(
                            "self._globalMessages[v[taskUUID]]: %s",
                            self._globalMessages[v["taskUUID"]],
                        )
                else:
                    self._globalMessages[value["taskUUID"]] = value

        def global_check(m):
            logger.debug("Global check message: %s", m)
            return accessDeepObject(taskUUID, m)

        logger.debug("Global Listener taskUUID: %s", taskUUID)

        temp_listener = self.addListener(check=global_check, lis=self._create_safe_async_listener(global_lis))
        logger.debug("globalListener :: Temp listener: %s", temp_listener)

        return temp_listener

    async def ensureConnection(self) -> None:
        """
        Ensure that a connection is established with the server.

        This method checks if the current connection is active and, if not, initiates a new connection.
        It handles authentication and retries the connection if necessary.

        :raises: An error message if the connection cannot be established due to an invalid API key or other reasons.
        """
        isConnected = self.connected() and self._ws.state is State.OPEN

        try:
            if self._invalidAPIkey:
                if not self._reconnection_manager._had_successful_auth:
                    raise ConnectionError(self._invalidAPIkey)

                circuit_state = self._reconnection_manager.get_state()
                if circuit_state == ConnectionState.CIRCUIT_OPEN:
                    raise ConnectionError(self._invalidAPIkey)

            if not isConnected:
                await self.connect()

                if self._invalidAPIkey and not self._reconnection_manager._had_successful_auth:
                    raise ConnectionError(self._invalidAPIkey)

        except Exception as e:
            raise ConnectionError(
                self._invalidAPIkey
                or "Could not connect to server. Ensure your API key is correct"
            )

    async def getSimililarImage(
        self,
        taskUUID: Union[str, List[str]],
        numberOfImages: int = 1,
        shouldThrowError: bool = True,
        lis: Optional[ListenerType] = None,
    ) -> List[IImage]:
        """
        Retrieve similar images based on the provided task UUID(s).

        :param taskUUID: A single task UUID or a list of task UUIDs.
        :param numberOfImages: The number of images to retrieve. Defaults to 1.
        :param shouldThrowError: Whether to raise an error on timeout. Defaults to True.
        :param lis: Optional listener to destroy upon completion.
        :param timeout: The timeout duration for the operation.
        :return: A list of IImage objects representing the images.
        """
        taskUUIDs = taskUUID if isinstance(taskUUID, list) else [taskUUID]

        async def check(
                resolve: Callable[[List[IImage]], None],
                reject: Callable[[IError], None],
                intervalId: Any,
        ) -> Optional[bool]:
            # Check if connection is currently lost
            if not self.connected() or not self.isWebsocketReadyState():
                reject(ConnectionError(
                    f"Connection lost while waiting for images | "
                    f"TaskUUIDs: {taskUUIDs}"
                ))
                return True

            async with self._images_lock:
                logger.debug(f"Check # Global images: {self._globalImages}")
                imagesWithSimilarTask = [
                    img
                    for img in self._globalImages
                    if img.get("taskType") in ["imageInference", "vectorize"]
                       and img.get("taskUUID") in taskUUIDs
                ]

                if self._globalError:
                    logger.debug(f"Check # _globalError: {self._globalError}")
                    error = self._globalError
                    self._globalError = None
                    logger.debug(f"Rejecting with error: {error}")
                    reject(RunwareError(error))
                    return True
                elif len(imagesWithSimilarTask) >= numberOfImages:
                    self._globalImages = [
                        img
                        for img in self._globalImages
                        if img.get("taskType") in ["imageInference", "vectorize"]
                           and img.get("taskUUID") not in taskUUIDs
                    ]
                    resolve(imagesWithSimilarTask[:numberOfImages])
                    return True

            return False

        try:
            return await getIntervalWithPromise(
                check,
                debugKey="getting images",
                shouldThrowError=shouldThrowError,
                timeOutDuration=IMAGE_INFERENCE_TIMEOUT,
            )
        except Exception as e:
            async with self._images_lock:
                current_images = len([
                    img for img in self._globalImages
                    if img.get("taskType") in ["imageInference", "vectorize"]
                       and img.get("taskUUID") in taskUUIDs
                ])
            error_msg = (
                f"Timeout waiting for images | "
                f"TaskUUIDs: {taskUUIDs} | "
                f"Expected: {numberOfImages} images | "
                f"Received: {current_images} images | "
                f"Timeout: {IMAGE_INFERENCE_TIMEOUT}ms | "
                f"Original error: {str(e)}"
            )
            raise Exception(error_msg) from e

    async def _modelUpload(
        self, requestModel: "IUploadModelBaseType"
    ) -> "Optional[IUploadModelResponse]":
        await self.ensureConnection()

        task_uuid = requestModel.taskUUID or getUUID()
        base_fields = {
            "taskType": ETaskType.MODEL_UPLOAD.value,
            "taskUUID": task_uuid,
            "air": requestModel.air,
            "name": requestModel.name,
            "downloadURL": requestModel.downloadURL,
            "uniqueIdentifier": requestModel.uniqueIdentifier,
            "version": requestModel.version,
            "format": requestModel.format,
            "private": requestModel.private,
            "category": requestModel.category,
            "architecture": requestModel.architecture,
        }

        optional_fields = [
            "retry",
            "heroImageURL",
            "tags",
            "shortDescription",
            "comment",
            "positiveTriggerWords",
            "type",
            "negativeTriggerWords",
            "defaultWeight",
            "defaultStrength",
            "defaultGuidanceScale",
            "defaultSteps",
            "defaultScheduler",
            "conditioning",
        ]

        request_object = {
            **base_fields,
            **{
                field: getattr(requestModel, field)
                for field in optional_fields
                if getattr(requestModel, field, None) is not None
            },
        }

        def is_ready(item: "Dict[str, Any]") -> bool:
            return item.get("status") == "ready"

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=is_ready,
            result_filter=lambda r: r.get("status") is not None
        )

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=MODEL_UPLOAD_TIMEOUT / 1000)

            unique_statuses = set()
            all_models = []

            for uploaded_model in results:
                if uploaded_model.get("code"):
                    self._handle_error_response(uploaded_model)

                status = uploaded_model.get("status")

                if status is not None and "error" in status:
                    self._handle_error_response(uploaded_model)

                if status not in unique_statuses:
                    all_models.append(uploaded_model)
                    unique_statuses.add(status)

            models = []
            for item in all_models:
                models.append(
                    {
                        "taskType": item.get("taskType"),
                        "taskUUID": item.get("taskUUID"),
                        "status": item.get("status"),
                        "message": item.get("message"),
                        "air": item.get("air"),
                    }
                )

            return models

        except asyncio.TimeoutError:
            op = self._pending_operations.get(task_uuid)
            partial_count = len(op["results"]) if op else 0
            raise Exception(
                f"Timeout waiting for model upload | TaskUUID: {task_uuid} | "
                f"Received: {partial_count} status updates | "
                f"Timeout: {MODEL_UPLOAD_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def modelUpload(
            self, requestModel: "IUploadModelBaseType"
    ) -> "Optional[IUploadModelResponse]":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._modelUpload, requestModel)

    async def modelSearch(self, payload: "IModelSearch") -> "IModelSearchResponse":
        async with self._request_semaphore:
            return await self._retry_with_reconnect(self._modelSearch, payload)

    async def _modelSearch(self, payload: "IModelSearch") -> "IModelSearchResponse":
        try:
            await self.ensureConnection()
            task_uuid = getUUID()

            request_object = {
                "taskUUID": task_uuid,
                "taskType": ETaskType.MODEL_SEARCH.value,
                **({"tags": payload.tags} if payload.tags else {}),
            }

            request_object.update(
                {
                    key: value
                    for key, value in vars(payload).items()
                    if value is not None and key != "additional_params"
                }
            )

            future, should_send = await self._register_pending_operation(
                task_uuid,
                expected_results=1,
                complete_predicate=lambda r: True
            )

            try:
                if should_send:
                    await self.send([request_object])
                    await self._mark_operation_sent(task_uuid)
                results = await asyncio.wait_for(future, timeout=self._timeout / 1000)
                response = results[0]
                self._handle_error_response(response)
                return instantiateDataclass(IModelSearchResponse, response)
            except asyncio.TimeoutError:
                raise Exception(
                    f"Timeout waiting for model search | TaskUUID: {task_uuid} | "
                    f"Timeout: {self._timeout}ms"
                )
            finally:
                await self._unregister_pending_operation(task_uuid)

        except RunwareAPIError:
            raise
        except Exception as e:
            if isinstance(e, RunwareAPIError):
                raise
            raise RunwareAPIError({"message": str(e)})

    async def videoInference(self, requestVideo: "IVideoInference") -> "Union[List[IVideo], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._videoInference,
                requestVideo,
                task_type=ETaskType.VIDEO_INFERENCE.value
            )

    async def _videoInference(self, requestVideo: IVideoInference) -> Union[List[IVideo], IAsyncTaskResponse]:
        await self.ensureConnection()
        return await self._requestVideo(requestVideo)

    async def inference3d(self, request3d: I3dInference) -> Union[List[I3d], IAsyncTaskResponse]:
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._inference3d,
                request3d,
                task_type=ETaskType.INFERENCE_3D.value,
            )

    async def _inference3d(self, request3d: I3dInference) -> Union[List[I3d], IAsyncTaskResponse]:
        await self.ensureConnection()
        return await self._request3d(request3d)

    async def textInference(self, requestText: ITextInference) -> Union[List[IText], IAsyncTaskResponse]:
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._requestText,
                requestText,
                task_type=ETaskType.TEXT_INFERENCE.value,
            )

    async def getResponse(
        self,
        taskUUID: str,
        numberResults: Optional[int] = 1,
    ) -> Union[List[IVideo], List[IAudio], List[IVideoToText], List[IImage], List[I3d], List[IText]]:
        async with self._request_semaphore:
            request = IGetResponseRequest(
                taskUUID=taskUUID,
                numberResults=numberResults or 1,
            )
            return await self._retry_with_reconnect(self._getResponse, request)

    async def _getResponse(
        self,
        request_model: IGetResponseRequest,
    ) -> Union[List[IVideo], List[IAudio], List[IVideoToText], List[IImage], List[I3d], List[IText]]:
        await self.ensureConnection()

        return await self._pollResults(
            task_uuid=request_model.taskUUID,
            number_results=request_model.numberResults,
        )

    async def _requestVideo(self, requestVideo: "IVideoInference") -> "Union[List[IVideo], IAsyncTaskResponse]":
        await self._processVideoImages(requestVideo)
        requestVideo.taskUUID = requestVideo.taskUUID or getUUID()
        request_object = self._buildVideoRequest(requestVideo)

        if requestVideo.webhookURL:
            request_object["webhookURL"] = requestVideo.webhookURL

        if requestVideo.skipResponse:
            await self.send([request_object])
            return IAsyncTaskResponse(
                taskType=ETaskType.VIDEO_INFERENCE.value,
                taskUUID=requestVideo.taskUUID
            )

        return await self._handleInitialVideoResponse(
            request_object=request_object,
            task_uuid=requestVideo.taskUUID,
            number_results=requestVideo.numberResults,
            delivery_method=requestVideo.deliveryMethod,
            webhook_url=request_object.get("webhookURL"),
            debug_key="video-inference-initial"
        )

    async def _processVideoImages(self, requestVideo: IVideoInference) -> None:
        frame_tasks = []
        reference_tasks = []

        if requestVideo.frameImages:
            frame_tasks = [
                process_image(frame_item.inputImage)
                for frame_item in requestVideo.frameImages
                if isinstance(frame_item, IFrameImage)
            ]

        if requestVideo.referenceImages:
            reference_tasks = [
                process_image(reference_item)
                for reference_item in requestVideo.referenceImages
            ]

        frame_results = await gather(*frame_tasks) if frame_tasks else []
        reference_results = await gather(*reference_tasks) if reference_tasks else []

        if requestVideo.frameImages and frame_results:
            processed_frame_images = []
            result_index = 0
            for frame_item in requestVideo.frameImages:
                if isinstance(frame_item, IFrameImage):
                    frame_item.inputImages = frame_results[result_index]
                    result_index += 1
                processed_frame_images.append(frame_item)
            requestVideo.frameImages = processed_frame_images

        if requestVideo.referenceImages and reference_results:
            requestVideo.referenceImages = reference_results

    def _buildVideoRequest(self, requestVideo: IVideoInference) -> Dict[str, Any]:
        request_object = {
            "deliveryMethod": requestVideo.deliveryMethod,
            "taskType": ETaskType.VIDEO_INFERENCE.value,
            "taskUUID": requestVideo.taskUUID,
            "model": requestVideo.model,
        }

        # Only add numberResults if it's not None
        if requestVideo.numberResults is not None:
            request_object["numberResults"] = requestVideo.numberResults

        # Only add positivePrompt if it's not None
        if requestVideo.positivePrompt is not None:
            request_object["positivePrompt"] = requestVideo.positivePrompt.strip()

        self._addOptionalField(request_object, requestVideo.speech)
        self._addOptionalVideoFields(request_object, requestVideo)
        self._addVideoImages(request_object, requestVideo)
        self._addOptionalField(request_object, requestVideo.inputs)
        self._addProviderSettings(request_object, requestVideo)
        self._addOptionalField(request_object, requestVideo.safety)
        self._addOptionalField(request_object, requestVideo.advancedFeatures)
        self._addOptionalField(request_object, requestVideo.acceleratorOptions)

        return request_object

    def _addOptionalVideoFields(self, request_object: Dict[str, Any], requestVideo: IVideoInference) -> None:
        optional_fields = [
            "outputType", "outputFormat", "outputQuality", "uploadEndpoint",
            "includeCost", "negativePrompt", "inputAudios", "referenceVideos", "fps", "steps", "scheduler", "seed",
            "CFGScale", "seedImage", "duration", "width", "height", "nsfw_check", "resolution",
        ]

        for field in optional_fields:
            value = getattr(requestVideo, field, None)
            if value is not None:
                request_object[field] = value

    def _addVideoImages(self, request_object: Dict[str, Any], requestVideo: IVideoInference) -> None:
        if requestVideo.frameImages:
            frame_images_data = []
            for frame_item in requestVideo.frameImages:
                frame_images_data.append({k: v for k, v in asdict(frame_item).items() if v is not None})
            request_object["frameImages"] = frame_images_data

        if requestVideo.referenceImages:
            request_object["referenceImages"] = requestVideo.referenceImages

        # Add lora if present
        if requestVideo.lora:
            request_object["lora"] = [
                {"model": lora.model, "weight": lora.weight}
                for lora in requestVideo.lora
            ]

    async def _process3dInputs(self, request3d: I3dInference) -> None:
        if not request3d.inputs:
            return
        if request3d.inputs.image:
            request3d.inputs.image = await process_image(request3d.inputs.image)
        if request3d.inputs.mask:
            request3d.inputs.mask = await process_image(request3d.inputs.mask)
        if request3d.inputs.meshFile:
            request3d.inputs.meshFile = await process_image(request3d.inputs.meshFile)

    def _build3dRequest(self, request3d: I3dInference) -> Dict[str, Any]:
        request_object: Dict[str, Any] = {
            "taskType": ETaskType.INFERENCE_3D.value,
            "taskUUID": request3d.taskUUID,
            "model": request3d.model,
        }
        if request3d.positivePrompt is not None:
            request_object["positivePrompt"] = request3d.positivePrompt.strip()
        if request3d.seed is not None:
            request_object["seed"] = request3d.seed
        if request3d.numberResults is not None:
            request_object["numberResults"] = request3d.numberResults
        if request3d.outputType is not None:
            request_object["outputType"] = request3d.outputType
        if request3d.outputFormat is not None:
            request_object["outputFormat"] = request3d.outputFormat
        if request3d.outputQuality is not None:
            request_object["outputQuality"] = request3d.outputQuality
        if request3d.includeCost is not None:
            request_object["includeCost"] = request3d.includeCost
        if request3d.deliveryMethod is not None:
            request_object["deliveryMethod"] = request3d.deliveryMethod
        if request3d.webhookURL is not None:
            request_object["webhookURL"] = request3d.webhookURL
        self._addOptionalField(request_object, request3d.inputs)
        self._addOptionalField(request_object, request3d.settings)

        return request_object

    async def _request3d(self, request3d: I3dInference) -> Union[List[I3d], IAsyncTaskResponse]:
        await self._process3dInputs(request3d)
        request3d.taskUUID = request3d.taskUUID or getUUID()
        request_object = self._build3dRequest(request3d)

        return await self._handleInitial3dResponse(
            request_object=request_object,
            task_uuid=request3d.taskUUID,
            number_results=request3d.numberResults or 1,
            delivery_method=request3d.deliveryMethod,
            webhook_url=request_object.get("webhookURL"),
            debug_key="3d-inference-initial",
        )

    def _buildTextRequest(self, requestText: ITextInference) -> Dict[str, Any]:
        request_object: Dict[str, Any] = {
            "taskType": ETaskType.TEXT_INFERENCE.value,
            "taskUUID": requestText.taskUUID,
            "model": requestText.model,
            "deliveryMethod": requestText.deliveryMethod,
            "messages": [asdict(m) for m in requestText.messages],
        }
        if requestText.maxTokens is not None:
            request_object["maxTokens"] = requestText.maxTokens
        if requestText.temperature is not None:
            request_object["temperature"] = requestText.temperature
        if requestText.topP is not None:
            request_object["topP"] = requestText.topP
        if requestText.topK is not None:
            request_object["topK"] = requestText.topK
        if requestText.seed is not None:
            request_object["seed"] = requestText.seed
        if requestText.stopSequences is not None:
            request_object["stopSequences"] = requestText.stopSequences
        if requestText.includeCost is not None:
            request_object["includeCost"] = requestText.includeCost
        self._addTextProviderSettings(request_object, requestText)
        return request_object

    async def _requestText(self, requestText: ITextInference) -> Union[List[IText], IAsyncTaskResponse]:
        await self.ensureConnection()
        requestText.taskUUID = requestText.taskUUID or getUUID()
        request_object = self._buildTextRequest(requestText)

        if requestText.webhookURL:
            request_object["webhookURL"] = requestText.webhookURL

        return await self._handleInitialTextResponse(
            request_object=request_object,
            task_uuid=requestText.taskUUID,
            number_results=requestText.numberResults or 1,
            delivery_method=requestText.deliveryMethod,
            webhook_url=request_object.get("webhookURL"),
            debug_key="text-inference-initial",
        )

    async def _handleInitialTextResponse(
        self,
        request_object: "Dict[str, Any]",
        task_uuid: str,
        number_results: int,
        delivery_method: "Union[str, EDeliveryMethod]" = EDeliveryMethod.SYNC,
        webhook_url: "Optional[str]" = None,
        debug_key: str = "text-inference-initial",
    ) -> Union[List[IText], IAsyncTaskResponse]:
        if delivery_method is None:
            delivery_method = EDeliveryMethod.SYNC
        delivery_method_enum = delivery_method if isinstance(delivery_method, EDeliveryMethod) else EDeliveryMethod(
            delivery_method,
        )

        def is_text_complete(r: "Dict[str, Any]") -> bool:
            if r.get("status") == "success":
                return True
            if r.get("text") is not None:
                return True
            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return True
            return False

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=is_text_complete,
        )

        timeout = TIMEOUT_DURATION if delivery_method_enum is EDeliveryMethod.SYNC else TEXT_INITIAL_TIMEOUT

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=timeout / 1000)

            if not results:
                raise ConnectionError(
                    f"No initial response received for text inference | "
                    f"delivery_method={delivery_method_enum} | taskUUID={task_uuid}"
                )

            response = results[0]
            self._handle_error_response(response)

            if response.get("status") == "success" or response.get("text") is not None:
                return instantiateDataclassList(IText, results)

            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return createAsyncTaskResponse(response)

            return instantiateDataclassList(IText, results)

        except asyncio.TimeoutError:
            if not self.connected() or not self.isWebsocketReadyState():
                raise ConnectionError(
                    f"Connection lost while waiting for text response | "
                    f"TaskUUID: {task_uuid} | Delivery method: {delivery_method_enum}"
                )

            if delivery_method_enum is EDeliveryMethod.SYNC:
                raise ConnectionError(
                    f"Timeout waiting for text generation | TaskUUID: {task_uuid} | "
                    f"Timeout: {timeout}ms"
                )
            raise ConnectionError(
                f"Timeout waiting for text generation | TaskUUID: {task_uuid} | "
                f"Timeout: {timeout}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    def _buildImageRequest(self, requestImage: IImageInference, prompt: Optional[str], control_net_data_dicts: List[Dict], instant_id_data: Optional[Dict], ip_adapters_data: Optional[List[Dict]], ace_plus_plus_data: Optional[Dict], pulid_data: Optional[Dict]) -> Dict[str, Any]:
        request_object = {
            "taskType": ETaskType.IMAGE_INFERENCE.value,
            "taskUUID": requestImage.taskUUID,
            "model": requestImage.model,
            "deliveryMethod": requestImage.deliveryMethod,
        }
        if prompt:
            request_object["positivePrompt"] = prompt

        self._addOptionalImageFields(request_object, requestImage)
        self._addImageSpecialFields(request_object, requestImage, control_net_data_dicts, instant_id_data, ip_adapters_data, ace_plus_plus_data, pulid_data)
        self._addOptionalField(request_object, requestImage.inputs)
        self._addImageProviderSettings(request_object, requestImage)
        self._addOptionalField(request_object, requestImage.ultralytics)
        self._addOptionalField(request_object, requestImage.safety)
        self._addOptionalField(request_object, requestImage.settings)


        return request_object

    def _addOptionalImageFields(self, request_object: Dict[str, Any], requestImage: IImageInference) -> None:
        optional_fields = [
            "outputType", "outputFormat", "outputQuality", "uploadEndpoint",
            "includeCost", "checkNsfw", "negativePrompt", "seedImage", "maskImage",
            "strength", "height", "width", "steps", "scheduler", "seed", "CFGScale",
            "clipSkip", "promptWeighting", "maskMargin", "vae", "webhookURL", "acceleration",
            "useCache", "ttl", "resolution"
        ]

        for field in optional_fields:
            value = getattr(requestImage, field, None)
            if value is not None:
                # Special handling for checkNsfw -> checkNSFW
                if field == "checkNsfw":
                    request_object["checkNSFW"] = value
                else:
                    request_object[field] = value

    def _addImageSpecialFields(self, request_object: Dict[str, Any], requestImage: IImageInference, control_net_data_dicts: List[Dict], instant_id_data: Optional[Dict], ip_adapters_data: Optional[List[Dict]], ace_plus_plus_data: Optional[Dict], pulid_data: Optional[Dict]) -> None:
        # Add controlNet if present
        if control_net_data_dicts:
            request_object["controlNet"] = control_net_data_dicts

        # Add lora if present
        if requestImage.lora:
            request_object["lora"] = [
                {"model": lora.model, "weight": lora.weight}
                for lora in requestImage.lora
            ]

        # Add lycoris if present
        if requestImage.lycoris:
            request_object["lycoris"] = [
                {"model": lycoris.model, "weight": lycoris.weight}
                for lycoris in requestImage.lycoris
            ]

        # Add embeddings if present
        if requestImage.embeddings:
            request_object["embeddings"] = [
                {"model": embedding.model}
                for embedding in requestImage.embeddings
            ]

        # Add refiner if present
        if requestImage.refiner:
            refiner_dict = {"model": requestImage.refiner.model}
            if requestImage.refiner.startStep is not None:
                refiner_dict["startStep"] = requestImage.refiner.startStep
            if requestImage.refiner.startStepPercentage is not None:
                refiner_dict["startStepPercentage"] = requestImage.refiner.startStepPercentage
            request_object["refiner"] = refiner_dict

        # Add instantID if present
        if instant_id_data:
            request_object["instantID"] = instant_id_data

        # Add outpaint if present
        if requestImage.outpaint:
            outpaint_dict = {
                k: v
                for k, v in vars(requestImage.outpaint).items()
                if v is not None
            }
            request_object["outpaint"] = outpaint_dict

        # Add ipAdapters if present
        if ip_adapters_data:
            request_object["ipAdapters"] = ip_adapters_data

        # Add acePlusPlus if present
        if ace_plus_plus_data:
            request_object["acePlusPlus"] = ace_plus_plus_data

        # Add puLID if present
        if pulid_data:
            request_object["puLID"] = pulid_data

        # Add referenceImages if present
        if requestImage.referenceImages:
            request_object["referenceImages"] = requestImage.referenceImages

        # Add acceleratorOptions if present
        self._addOptionalField(request_object, requestImage.acceleratorOptions)

        # Add advancedFeatures if present
        if requestImage.advancedFeatures:
            pipeline_options_dict = {
                k: v.__dict__
                for k, v in vars(requestImage.advancedFeatures).items()
                if v is not None
            }
            request_object["advancedFeatures"] = pipeline_options_dict

        # Add extraArgs if present
        if hasattr(requestImage, "extraArgs") and isinstance(requestImage.extraArgs, dict):
            request_object.update(requestImage.extraArgs)

    def _addSafetySettings(self, request_object: Dict[str, Any], safety: ISafety) -> None:
        safety_dict = asdict(safety)
        safety_dict = {k: v for k, v in safety_dict.items() if v is not None}
        if safety_dict:
            request_object["safety"] = safety_dict

    def _addImageProviderSettings(self, request_object: Dict[str, Any], requestImage: IImageInference) -> None:
        if not requestImage.providerSettings:
            return
        provider_dict = requestImage.providerSettings.to_request_dict()
        if provider_dict:
            request_object["providerSettings"] = provider_dict

    def _addProviderSettings(self, request_object: Dict[str, Any], requestVideo: IVideoInference) -> None:
        if not requestVideo.providerSettings:
            return
        provider_dict = requestVideo.providerSettings.to_request_dict()
        if provider_dict:
            request_object["providerSettings"] = provider_dict

    def _addOptionalField(self, request_object: Dict[str, Any], obj: Any) -> None:
        if not obj:
            return
        obj_dict = obj.to_request_dict()
        if obj_dict:
            request_object.update(obj_dict)

    async def _handleWebhookRequest(
        self,
        request_object: "Dict[str, Any]",
        task_uuid: str,
        task_type: str,
        debug_key: str,
    ) -> "IAsyncTaskResponse":
        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=lambda r: r.get("taskType") == task_type or r.get("taskUUID") == task_uuid
        )

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=WEBHOOK_TIMEOUT / 1000)
            response = results[0]
            self._handle_error_response(response)
            return createAsyncTaskResponse(response)
        except asyncio.TimeoutError:
            raise Exception(
                f"Timeout waiting for webhook acknowledgment | TaskUUID: {task_uuid} | "
                f"TaskType: {task_type} | Timeout: {WEBHOOK_TIMEOUT}ms"
            )
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def _handleInitialVideoResponse(
            self,
            request_object: "Dict[str, Any]",
            task_uuid: str,
            number_results: int,
            delivery_method: "Union[str, EDeliveryMethod]" = None,
            webhook_url: "Optional[str]" = None,
            debug_key: str = "video-inference-initial"
    ) -> "Union[List[IVideo], IAsyncTaskResponse]":
        if delivery_method is None:
            delivery_method = EDeliveryMethod.ASYNC
        delivery_method_enum = delivery_method if isinstance(delivery_method, EDeliveryMethod) else EDeliveryMethod(
            delivery_method)

        def is_video_complete(r: "Dict[str, Any]") -> bool:
            if r.get("status") == "success":
                return True
            if r.get("videoUUID") is not None or r.get("mediaUUID") is not None:
                return True
            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return True
            return False

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=is_video_complete
        )

        timeout = TIMEOUT_DURATION if delivery_method_enum is EDeliveryMethod.SYNC else VIDEO_INITIAL_TIMEOUT

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=timeout / 1000)

            if not results:
                raise ConnectionError(
                    f"No initial response received for video generation | "
                    f"delivery_method={delivery_method_enum} | taskUUID={task_uuid}"
                )

            response = results[0]
            self._handle_error_response(response)

            if response.get("status") == "success" or response.get("videoUUID") is not None or response.get(
                    "mediaUUID") is not None:
                return instantiateDataclassList(IVideo, results)

            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return createAsyncTaskResponse(response)

            return instantiateDataclassList(IVideo, results)

        except asyncio.TimeoutError:
            if not self.connected() or not self.isWebsocketReadyState():
                raise ConnectionError(
                    f"Connection lost while waiting for video response | "
                    f"TaskUUID: {task_uuid} | Delivery method: {delivery_method_enum}"
                )

            if delivery_method_enum is EDeliveryMethod.SYNC:
                raise ConnectionError(
                    f"Timeout waiting for video generation | TaskUUID: {task_uuid} | "
                    f"Timeout: {timeout}ms"
                )
            raise ConnectionError(
                f"Timeout waiting for video generation | TaskUUID: {task_uuid} | "
                f"Timeout: {timeout}ms"
            )
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def _handleInitial3dResponse(
        self,
        request_object: Dict[str, Any],
        task_uuid: str,
        number_results: int,
        delivery_method: Union[str, EDeliveryMethod] = EDeliveryMethod.ASYNC,
        webhook_url: Optional[str] = None,
        debug_key: str = "3d-inference-initial",
    ) -> Union[List[I3d], IAsyncTaskResponse]:
        delivery_method_enum = delivery_method if isinstance(delivery_method, EDeliveryMethod) else EDeliveryMethod(delivery_method)

        def is_3d_complete(r: Dict[str, Any]) -> bool:
            if r.get("status") == "success":
                return True
            outputs = r.get("outputs")
            if outputs is not None and outputs.get("files") is not None:
                return True
            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return True
            return False

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=is_3d_complete,
        )

        timeout = TIMEOUT_DURATION if delivery_method_enum is EDeliveryMethod.SYNC else VIDEO_INITIAL_TIMEOUT

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=timeout / 1000)

            if not results:
                raise ConnectionError(
                    f"No initial response received for 3d generation | "
                    f"delivery_method={delivery_method_enum} | taskUUID={task_uuid}"
                )

            response = results[0]
            self._handle_error_response(response)

            if response.get("status") == "success":
                return instantiateDataclassList(I3d, results)
            outputs = response.get("outputs")
            if outputs is not None and outputs.get("files") is not None:
                return instantiateDataclassList(I3d, results)

            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return createAsyncTaskResponse(response)

            return instantiateDataclassList(I3d, results)

        except asyncio.TimeoutError:
            if not self.connected() or not self.isWebsocketReadyState():
                raise ConnectionError(
                    f"Connection lost while waiting for 3d response | "
                    f"TaskUUID: {task_uuid} | Delivery method: {delivery_method_enum}"
                )

            if delivery_method_enum is EDeliveryMethod.SYNC:
                raise ConnectionError(
                    f"Timeout waiting for 3d generation | TaskUUID: {task_uuid} | "
                    f"Timeout: {timeout}ms"
                )
            raise ConnectionError(
                f"Timeout waiting for 3d generation | TaskUUID: {task_uuid} | "
                f"Timeout: {timeout}ms"
            )
        finally:
            await self._unregister_pending_operation(task_uuid)

    async def _handleInitialImageResponse(
        self,
        task_uuid: str,
        number_results: int,
        delivery_method: Union[str, EDeliveryMethod] = EDeliveryMethod.ASYNC,
        webhook_url: Optional[str] = None,
        debug_key: str = "image-inference-initial"
    ) -> Union[List[IImage], IAsyncTaskResponse]:
        lis = self.globalListener(taskUUID=task_uuid)
        delivery_method_enum = delivery_method if isinstance(delivery_method, EDeliveryMethod) else EDeliveryMethod(delivery_method)

        async def check_initial_response(resolve: callable, reject: callable, *args: Any) -> bool:
            # Check if connection was lost during the wait
            if not self.connected() or not self.isWebsocketReadyState():
                reject(ConnectionError(
                    f"Connection lost while waiting for image response | "
                    f"TaskUUID: {task_uuid} | "
                    f"Delivery method: {delivery_method_enum}"
                ))
                return True
            
            async with self._messages_lock:
                response_list = self._globalMessages.get(task_uuid, [])

                if not response_list:
                    return False

                response = response_list[0]

                if self._is_error_response(response):
                    del self._globalMessages[task_uuid]
                    raise RunwareAPIError(response)

                if response.get("status") == "success" or response.get("imageUUID") is not None:
                    del self._globalMessages[task_uuid]
                    resolve([response])
                    return True

                if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                    del self._globalMessages[task_uuid]
                    async_response = createAsyncTaskResponse(response)
                    resolve([async_response])
                    return True

                return False

        try:
            initial_response = await getIntervalWithPromise(
                check_initial_response,
                debugKey=debug_key,
                timeOutDuration=TIMEOUT_DURATION if delivery_method_enum is EDeliveryMethod.SYNC else IMAGE_INITIAL_TIMEOUT
            )
        except RunwareAPIError:
            raise
        except Exception as e:
            if not self.connected() or not self.isWebsocketReadyState():
                raise ConnectionError(
                    f"Connection lost while waiting for image response | "
                    f"TaskUUID: {task_uuid} | "
                    f"Delivery method: {delivery_method_enum}"
                )
            if delivery_method_enum is EDeliveryMethod.SYNC:
                error_msg = (
                    f"Timeout waiting for image generation | "
                    f"TaskUUID: {task_uuid} | "
                    f"Timeout: {TIMEOUT_DURATION}ms | "
                    f"Original error: {str(e)}"
                )
                raise ConnectionError(error_msg)
            initial_response = None
        finally:
            lis["destroy"]()

        if not initial_response or len(initial_response) == 0:
            raise ConnectionError(
                f"No initial response received for image generation | delivery_method={delivery_method_enum} | taskUUID={task_uuid}"
            )

        if isinstance(initial_response[0], IAsyncTaskResponse):
            return initial_response[0]

        return instantiateDataclassList(IImage, initial_response)

    async def _sendPollRequest(self, task_uuid: str, poll_count: int) -> "List[Dict[str, Any]]":
        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=lambda r: True
        )
        try:
            if should_send:
                await self.send([{
                    "taskType": ETaskType.GET_RESPONSE.value,
                    "taskUUID": task_uuid
                }])
                await self._mark_operation_sent(task_uuid)

            results = await asyncio.wait_for(future, timeout=VIDEO_INITIAL_TIMEOUT / 1000)
            return results

        except asyncio.TimeoutError:
            op = self._pending_operations.get(task_uuid)
            if op and op["results"]:
                return op["results"]
            raise Exception(
                f"Timeout waiting for poll response | TaskUUID: {task_uuid} | "
                f"Poll: {poll_count} | Timeout: {VIDEO_INITIAL_TIMEOUT}ms"
            )
        finally:
            await self._unregister_pending_operation(task_uuid)

    def _hasPendingResults(self, responses: List[Dict[str, Any]]) -> bool:
        return any(response.get("status") == "processing" for response in responses)

    async def audioInference(self, requestAudio: "IAudioInference") -> "Union[List[IAudio], IAsyncTaskResponse]":
        async with self._request_semaphore:
            return await self._retry_async_with_reconnect(
                self._requestAudio,
                requestAudio,
                task_type=ETaskType.AUDIO_INFERENCE.value
            )

    async def _requestAudio(self, requestAudio: "IAudioInference") -> Union[List["IAudio"], "IAsyncTaskResponse"]:
        await self.ensureConnection()
        requestAudio.taskUUID = requestAudio.taskUUID or getUUID()
        request_object = self._buildAudioRequest(requestAudio)

        return await self._handleInitialAudioResponse(
            request_object=request_object,
            task_uuid=requestAudio.taskUUID,
            number_results=requestAudio.numberResults,
            delivery_method=requestAudio.deliveryMethod,
            webhook_url=request_object.get("webhookURL"),
            debug_key="audio-inference-initial"
        )

    def _buildAudioRequest(self, requestAudio: IAudioInference) -> Dict[str, Any]:
        request_object = {
            "deliveryMethod": requestAudio.deliveryMethod,
            "taskType": ETaskType.AUDIO_INFERENCE.value,
            "taskUUID": requestAudio.taskUUID,
            "model": requestAudio.model,
            "numberResults": requestAudio.numberResults,
        }

        # Only add positivePrompt if it's provided
        if requestAudio.positivePrompt is not None:
            request_object["positivePrompt"] = requestAudio.positivePrompt.strip()

        # Only add duration if it's provided and not using composition plan
        if requestAudio.duration is not None:
            request_object["duration"] = requestAudio.duration

        self._addOptionalAudioFields(request_object, requestAudio)
        self._addOptionalField(request_object, requestAudio.speech)
        self._addOptionalField(request_object, requestAudio.audioSettings)
        self._addOptionalField(request_object, requestAudio.settings)
        self._addAudioProviderSettings(request_object, requestAudio)
        self._addOptionalField(request_object, requestAudio.inputs)
        self._addOptionalField(request_object, requestAudio.settings)
        
        return request_object

    def _addOptionalAudioFields(self, request_object: Dict[str, Any], requestAudio: IAudioInference) -> None:
        optional_fields = [
            "outputType", "outputFormat", "includeCost", "uploadEndpoint", "webhookURL",
            "negativePrompt", "steps", "seed", "CFGScale", "strength"
        ]

        for field in optional_fields:
            value = getattr(requestAudio, field, None)
            if value is not None:
                request_object[field] = value


    def _addAudioProviderSettings(self, request_object: Dict[str, Any], requestAudio: IAudioInference) -> None:
        if not requestAudio.providerSettings:
            return
        provider_dict = requestAudio.providerSettings.to_request_dict()
        if provider_dict:
            request_object["providerSettings"] = provider_dict

    def _addTextProviderSettings(self, request_object: Dict[str, Any], requestText: ITextInference) -> None:
        if not requestText.providerSettings:
            return
        provider_dict = requestText.providerSettings.to_request_dict()
        if provider_dict:
            request_object["providerSettings"] = provider_dict

    async def _handleInitialAudioResponse(
            self,
            request_object: "Dict[str, Any]",
            task_uuid: str,
            number_results: int,
            delivery_method: "Union[str, EDeliveryMethod]" = None,
            webhook_url: "Optional[str]" = None,
            debug_key: str = "audio-inference-initial"
    ) -> "Union[List[IAudio], IAsyncTaskResponse]":
        if delivery_method is None:
            delivery_method = EDeliveryMethod.SYNC
        delivery_method_enum = delivery_method if isinstance(delivery_method, EDeliveryMethod) else EDeliveryMethod(
            delivery_method)

        def is_audio_complete(r: "Dict[str, Any]") -> bool:
            if r.get("status") == "success":
                return True
            if r.get("audioUUID") is not None:
                return True
            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return True
            return False

        future, should_send = await self._register_pending_operation(
            task_uuid,
            expected_results=1,
            complete_predicate=is_audio_complete
        )

        timeout = TIMEOUT_DURATION if delivery_method_enum is EDeliveryMethod.SYNC else AUDIO_INITIAL_TIMEOUT

        try:
            if should_send:
                await self.send([request_object])
                await self._mark_operation_sent(task_uuid)
            results = await asyncio.wait_for(future, timeout=timeout / 1000)

            if not results:
                raise ConnectionError(
                    f"No initial response received for audio inference | "
                    f"delivery_method={delivery_method_enum} | taskUUID={task_uuid}"
                )

            response = results[0]
            self._handle_error_response(response)

            if response.get("status") == "success" or response.get("audioUUID") is not None:
                return instantiateDataclassList(IAudio, results)

            if webhook_url or delivery_method_enum is EDeliveryMethod.ASYNC:
                return createAsyncTaskResponse(response)

            return instantiateDataclassList(IAudio, results)

        except asyncio.TimeoutError:
            if not self.connected() or not self.isWebsocketReadyState():
                raise ConnectionError(
                    f"Connection lost while waiting for audio response | "
                    f"TaskUUID: {task_uuid} | Delivery method: {delivery_method_enum}"
                )

            if delivery_method_enum is EDeliveryMethod.SYNC:
                raise ConnectionError(
                    f"Timeout waiting for audio generation | TaskUUID: {task_uuid} | "
                    f"Timeout: {timeout}ms"
                )
            raise
        except RunwareAPIError:
            raise
        finally:
            await self._unregister_pending_operation(task_uuid)

    def _processPollingResponse(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        completed_results: List[Dict[str, Any]] = []

        for response in responses:
            self._handle_error_response(response)

            if response.get("status") == "success":
                completed_results.append(response)

        return completed_results

    async def _pollResults(
        self,
        task_uuid: str,
        number_results: Optional[int],
    ) -> Union[List[IVideo], List[IVideoToText], List[IAudio], List[IImage], List[I3d], List[IText]]:
        # Default to 1 if number_results is None
        if number_results is None:
            number_results = 1

        completed_results: "List[Dict[str, Any]]" = []

        task_type = None
        response_cls: Optional[Union[IVideo, IVideoToText, IAudio, IImage, I3d, IText]] = None
        max_polls: int = MAX_POLLS
        polling_delay: int = VIDEO_POLLING_DELAY
        timeout_message: str = f"Polling timeout after {MAX_POLLS} polls"

        def configure_from_task_type(task_type_val: Optional[str]):
            if not task_type_val:
                return None

            match task_type_val:
                case ETaskType.AUDIO_INFERENCE.value:
                    return (
                        IAudio,
                        MAX_POLLS_AUDIO_GENERATION,
                        AUDIO_POLLING_DELAY,
                        f"Audio generation timeout after {MAX_POLLS_AUDIO_GENERATION} polls"
                    )
                case ETaskType.VIDEO_CAPTION.value:
                    return (
                        IVideoToText,
                        MAX_POLLS_VIDEO_GENERATION,
                        VIDEO_POLLING_DELAY,
                        f"Video caption generation timeout after {MAX_POLLS_VIDEO_GENERATION} polls"
                    )
                case ETaskType.IMAGE_INFERENCE.value:
                    return (
                        IImage,
                        MAX_POLLS_IMAGE_GENERATION,
                        IMAGE_POLLING_DELAY,
                        f"Image generation timeout after {MAX_POLLS_IMAGE_GENERATION} polls"
                    )
                case (
                    ETaskType.VIDEO_INFERENCE.value
                    | ETaskType.VIDEO_BACKGROUND_REMOVAL.value
                    | ETaskType.VIDEO_UPSCALE.value
                ):
                    return (
                        IVideo,
                        MAX_POLLS_VIDEO_GENERATION,
                        VIDEO_POLLING_DELAY,
                        f"Video generation timeout after {MAX_POLLS_VIDEO_GENERATION} polls"
                    )
                case ETaskType.INFERENCE_3D.value:
                    return (
                        I3d,
                        MAX_POLLS_3D_GENERATION,
                        VIDEO_POLLING_DELAY,
                        f"3d generation timeout after {MAX_POLLS_3D_GENERATION} polls"
                    )
                case ETaskType.TEXT_INFERENCE.value:
                    return (
                        IText,
                        MAX_POLLS,
                        TEXT_POLLING_DELAY,
                        f"Text generation timeout after {MAX_POLLS} polls"
                    )
                case _:
                    raise ValueError(f"Unsupported task type for polling: {task_type_val}")

        max_polls_loop = max(
            MAX_POLLS,
            MAX_POLLS_VIDEO_GENERATION,
            MAX_POLLS_AUDIO_GENERATION,
            MAX_POLLS_3D_GENERATION,
            MAX_POLLS_IMAGE_GENERATION,
        )
        try:
            for poll_count in range(max_polls_loop):
                try:
                    responses = await self._sendPollRequest(task_uuid, poll_count)
                    
                    for response in responses:
                        self._handle_error_response(response)

                    if task_type is None:
                        for resp in responses:
                            task_type = resp.get("taskType")
                            if task_type:
                                response_config = configure_from_task_type(task_type)
                                if response_config:
                                    response_cls, max_polls, polling_delay, timeout_message = response_config
                                break

                    processed_responses = self._processPollingResponse(responses)
                    completed_results.extend(processed_responses)

                    if len(completed_results) >= number_results:

                        return instantiateDataclassList(
                            response_cls or IVideo,
                            completed_results[:number_results]
                        )

                    has_pending = self._hasPendingResults(responses)
                    has_queued = any(
                        response.get("status") in ("queued", "pending", "scheduled", "waiting")
                        for response in responses
                    )

                    if not processed_responses and not has_pending and not has_queued:
                        has_task_response = any(r.get("taskUUID") == task_uuid for r in responses)
                        if has_task_response:
                            logger.warning(f"Received response for {task_uuid} but status unclear, continuing poll")
                            continue
                        raise RunwareAPIError({"message": f"Unexpected polling response at poll {poll_count}"})

                except RunwareAPIError:
                    raise
                except Exception as e:
                    if poll_count >= max_polls - 1:
                        raise e

                if poll_count >= max_polls - 1:
                    raise RunwareAPIError({"message": timeout_message})

                await asyncio.sleep(polling_delay / 1000)

            if completed_results:
                return instantiateDataclassList(response_cls or IVideo, completed_results[:number_results])
            return []

        except Exception:
            raise

    def _createAudioFromResponse(self, response: Dict[str, Any]) -> IAudio:
        return IAudio(
            taskType=response.get("taskType", ""),
            taskUUID=response.get("taskUUID", ""),
            status=response.get("status"),
            audioUUID=response.get("audioUUID"),
            audioURL=response.get("audioURL"),
            audioBase64Data=response.get("audioBase64Data"),
            audioDataURI=response.get("audioDataURI"),
            cost=response.get("cost")
        )

    def connected(self) -> bool:
        """
        Check if the current WebSocket connection is active and authenticated.

        :return: True if the connection is active and authenticated, False otherwise.
        """
        return self.isWebsocketReadyState() and self._connectionSessionUUID is not None

    def _is_error_response(self, response: Dict[str, Any]) -> bool:
        """Check if response indicates an error via 'code', 'error', or 'errorId' fields."""
        if not response or not isinstance(response, dict):
            return False

        if response.get("code"):
            return True
        if response.get("error"):
            return True
        if response.get("errorId"):
            return True

        return False
