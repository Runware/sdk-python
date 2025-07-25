import asyncio
import atexit
import weakref
from typing import Any, Callable, Dict, List, Optional, Union

from .connection.manager import ConnectionManager, ConnectionState
from .core.types import ProgressUpdate
from .exceptions import RunwareAuthenticationError
from .logging_config import get_logger, setup_logging
from .messaging.router import MessageRouter
from .operations.image_background_removal import ImageBackgroundRemovalOperation
from .operations.image_caption import ImageCaptionOperation
from .operations.image_inference import ImageInferenceOperation
from .operations.image_upscale import ImageUpscaleOperation
from .operations.manager import OperationManager
from .operations.model_search import ModelSearchOperation
from .operations.model_upload import ModelUploadOperation
from .operations.photo_maker import PhotoMakerOperation
from .operations.prompt_enhance import PromptEnhanceOperation
from .operations.video_inference import VideoInferenceOperation
from .types import (
    Environment,
    File,
    IEnhancedPrompt,
    IImage,
    IImageBackgroundRemoval,
    IImageCaption,
    IImageInference,
    IImageToText,
    IImageUpscale,
    IModelSearch,
    IModelSearchResponse,
    IPhotoMaker,
    IPromptEnhance,
    IUploadModelBaseType,
    IUploadModelResponse,
    IVideo,
    IVideoInference,
    UploadImageType,
)
from .utils import BASE_RUNWARE_URLS, fileToBase64, isLocalFile

logger = get_logger(__name__)
_active_clients = weakref.WeakSet()


def _cleanup_all_clients():
    """Cleanup for all active clients"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return
    except RuntimeError:
        return
    for client in list(_active_clients):
        try:
            client._force_sync_cleanup()
        except Exception:
            pass


atexit.register(_cleanup_all_clients)


class RunwareClient:
    """
    Runware API client for AI image and video generation.

    This client provides a high-level interface for interacting with the Runware API,
    supporting various AI operations like image generation, video generation, and more.
    """

    def __init__(
        self,
        api_key: str,
        url: str = BASE_RUNWARE_URLS[Environment.PRODUCTION],
        max_concurrent_operations: int = 50,
        default_timeout: float = 300.0,
        log_level: str = "CRITICAL",
        auto_disconnect: bool = True,
        # New parameters for operation-specific timeouts
        video_timeout: Optional[float] = None,
        image_timeout: Optional[float] = None,
        # Legacy parameters for backward compatibility
        timeout: Optional[int] = None,
    ):
        """
        Initialize the Runware client.

        Args:
            api_key: Your Runware API key
            url: The Runware API endpoint URL
            max_concurrent_operations: Maximum number of concurrent operations
            default_timeout: Default timeout for operations in seconds
            log_level: Logging level for the SDK
            auto_disconnect: Whether to auto-disconnect after operations complete
            video_timeout: Specific timeout for video operations
            image_timeout: Specific timeout for image operations
            timeout: Legacy timeout parameter (converted from ms to seconds)
        """
        self.api_key = api_key
        self.url = url

        # Handle legacy timeout parameter
        if timeout is not None:
            self.default_timeout = timeout / 1000.0  # Convert ms to seconds
        else:
            self.default_timeout = default_timeout

        # Set operation-specific timeouts
        self.video_timeout = video_timeout or 1800.0  # 30 minutes for video operations
        self.image_timeout = image_timeout or default_timeout  # Use default for images

        self.auto_disconnect = auto_disconnect

        # Setup logging
        setup_logging(log_level)
        self.logger = get_logger(self.__class__.__name__)

        # Initialize core components
        self.message_router = MessageRouter()
        self.operation_manager = OperationManager(
            max_concurrent_operations=max_concurrent_operations,
            operation_timeout=self.default_timeout,
        )
        self.connection_manager = ConnectionManager(
            api_key=api_key, url=url, message_router=self.message_router
        )

        # Client state management
        self._is_started = False
        self._is_disconnecting = False
        self._connection_callbacks: List[Callable[[ConnectionState], None]] = []

        # Register for cleanup
        _active_clients.add(self)

        self.logger.info(f"RunwareClient initialized with URL: {url}")
        self.logger.info(
            f"Timeouts - Default: {self.default_timeout}s, Video: {self.video_timeout}s, Image: {self.image_timeout}s"
        )

    def __del__(self):
        """Cleanup method - will attempt to disconnect synchronously if possible"""
        if self._is_started and not self._is_disconnecting:
            self._force_sync_cleanup()

    def _force_sync_cleanup(self):
        """Force synchronous cleanup of resources"""
        if self._is_disconnecting:
            return

        self._is_disconnecting = True

        try:
            # Try to get current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    loop.create_task(self._emergency_disconnect())
                    return
            except RuntimeError:
                pass

            self._is_started = False

        except Exception:
            # Silently ignore cleanup errors in destructor
            pass

    async def _emergency_disconnect(self):
        """Emergency async disconnect"""
        try:
            await asyncio.wait_for(self.disconnect(), timeout=5.0)
        except Exception:
            self._force_sync_cleanup()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    # Legacy method for backward compatibility
    async def ensureConnection(self):
        """Legacy method - use connect() instead"""
        if not self._is_started:
            await self.connect()

    async def connect(self):
        """Establish connection to the Runware API"""
        if self._is_started:
            self.logger.warning("Client already started")
            return

        self.logger.info("Starting Runware client")

        try:
            await self.message_router.start()
            await self.operation_manager.start()

            self.connection_manager.add_connection_callback(
                self._on_connection_state_change
            )
            await self.connection_manager.start()

            authenticated = await self.connection_manager.wait_for_authentication(
                timeout=30.0
            )
            if not authenticated:
                raise RunwareAuthenticationError(
                    "Failed to authenticate within timeout"
                )

            self._is_started = True
            self.logger.info("Runware client started successfully")

        except Exception as e:
            self.logger.error("Failed to start Runware client", exc_info=e)
            await self._cleanup()
            raise

    async def disconnect(self):
        """Disconnect from the Runware API"""
        if not self._is_started or self._is_disconnecting:
            return

        self._is_disconnecting = True
        self.logger.info("Starting Runware client shutdown")

        try:
            # Stop components in reverse order with timeouts
            await asyncio.wait_for(self.operation_manager.stop(), timeout=10.0)
            await asyncio.wait_for(self.message_router.stop(), timeout=10.0)
            await asyncio.wait_for(self.connection_manager.stop(), timeout=10.0)

            self._is_started = False
            self.logger.info("Runware client stopped successfully")

        except asyncio.TimeoutError as e:
            self.logger.error("Timeout during client shutdown", exc_info=e)
        except Exception as e:
            self.logger.error("Error during client shutdown", exc_info=e)
        finally:
            self._is_disconnecting = False

    def is_connected(self) -> bool:
        """Check if the client is connected and authenticated"""
        return self._is_started and self.connection_manager.is_authenticated()

    # Legacy methods for backward compatibility
    def connected(self) -> bool:
        """Legacy method - use is_connected() instead"""
        return self.is_connected()

    def isWebsocketReadyState(self) -> bool:
        """Legacy method - use is_connected() instead"""
        return self.is_connected()

    def isAuthenticated(self) -> bool:
        """Legacy method - use is_connected() instead"""
        return self.is_connected()

    def add_connection_callback(self, callback: Callable[[ConnectionState], None]):
        """Add a callback for connection state changes"""
        self._connection_callbacks.append(callback)
        if self._is_started:
            self.connection_manager.add_connection_callback(callback)

    def remove_connection_callback(self, callback: Callable[[ConnectionState], None]):
        """Remove a connection state change callback"""
        if callback in self._connection_callbacks:
            self._connection_callbacks.remove(callback)
        if self._is_started:
            self.connection_manager.remove_connection_callback(callback)

    def _get_operation_timeout(
        self, operation_type: str, user_timeout: Optional[float] = None
    ) -> float:
        """Get appropriate timeout for operation type"""
        if user_timeout is not None:
            return user_timeout

        if operation_type == "videoInference":
            return self.video_timeout
        elif operation_type in [
            "imageInference",
            "photoMaker",
            "imageCaption",
            "imageBackgroundRemoval",
            "imageUpscale",
        ]:
            return self.image_timeout
        else:
            return self.default_timeout

    async def _execute_operation(self, operation, operation_timeout: float):
        """
        Common operation execution logic using Template Method pattern.

        This method encapsulates the common steps for executing any operation:
        1. Register the operation with the message router
        2. Execute the operation with the operation manager
        3. Unregister the operation
        4. Handle auto-disconnect if configured
        """
        await self.message_router.register_operation(operation)

        try:
            results = await self.operation_manager.execute_operation(
                operation, operation_timeout
            )
            return results
        finally:
            await self.message_router.unregister_operation(operation.operation_id)
            await self._check_auto_disconnect()

    async def imageInference(
        self,
        requestImage: IImageInference,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> List[IImage]:
        """
        Generate images using AI image inference.

        Args:
            requestImage: Image inference request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of generated images
        """
        await self.ensureConnection()

        operation_timeout = self._get_operation_timeout("imageInference", timeout)
        self.logger.debug(f"Using timeout {operation_timeout}s for image inference")

        operation = ImageInferenceOperation(requestImage, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def videoInference(
        self,
        requestVideo: IVideoInference,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> List[IVideo]:
        """
        Generate videos using AI video inference.

        Args:
            requestVideo: Video inference request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of generated videos
        """
        await self.ensureConnection()

        operation_timeout = self._get_operation_timeout("videoInference", timeout)
        self.logger.info(
            f"Starting video inference with timeout {operation_timeout}s ({operation_timeout / 60:.1f} minutes)"
        )

        operation = VideoInferenceOperation(requestVideo, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        """
        Upload an image file or return existing image reference.

        Args:
            file: File object, file path, or existing image reference

        Returns:
            UploadImageType with image UUID and URL
        """
        await self.ensureConnection()

        if isinstance(file, str):
            if not isLocalFile(file):
                return UploadImageType(imageUUID=file, imageURL=file, taskUUID="direct")

            try:
                file = await fileToBase64(file)
            except Exception as e:
                raise

        return UploadImageType(
            imageUUID=str(hash(file))[:16], imageURL="uploaded", taskUUID="upload"
        )

    async def imageCaption(
        self,
        requestImageToText: IImageCaption,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> IImageToText:
        """
        Generate captions for images.

        Args:
            requestImageToText: Image caption request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of image captions
        """
        await self.ensureConnection()

        operation_timeout = self._get_operation_timeout("imageCaption", timeout)

        operation = ImageCaptionOperation(requestImageToText, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def imageBackgroundRemoval(
        self,
        removeImageBackgroundPayload: IImageBackgroundRemoval,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> List[IImage]:
        """
        Remove backgrounds from images.

        Args:
            removeImageBackgroundPayload: Background removal request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of images with removed backgrounds
        """
        await self.ensureConnection()

        operation_timeout = self._get_operation_timeout(
            "imageBackgroundRemoval", timeout
        )

        operation = ImageBackgroundRemovalOperation(removeImageBackgroundPayload, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def imageUpscale(
        self,
        upscaleGanPayload: IImageUpscale,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> List[IImage]:
        """
        Upscale images using AI.

        Args:
            upscaleGanPayload: Image upscale request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of upscaled images
        """
        await self.ensureConnection()

        operation_timeout = self._get_operation_timeout("imageUpscale", timeout)

        operation = ImageUpscaleOperation(upscaleGanPayload, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def promptEnhance(
        self,
        promptEnhancer: IPromptEnhance,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> List[IEnhancedPrompt]:
        """
        Enhance prompts using AI.

        Args:
            promptEnhancer: Prompt enhancement request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of enhanced prompts
        """
        await self.ensureConnection()

        operation_timeout = timeout or self.default_timeout

        operation = PromptEnhanceOperation(promptEnhancer, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def photoMaker(
        self,
        requestPhotoMaker: IPhotoMaker,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> List[IImage]:
        """
        Generate images with PhotoMaker.

        Args:
            requestPhotoMaker: PhotoMaker request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of generated images
        """
        await self.ensureConnection()

        operation_timeout = self._get_operation_timeout("photoMaker", timeout)

        operation = PhotoMakerOperation(requestPhotoMaker, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def modelUpload(
        self,
        requestModel: IUploadModelBaseType,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> Optional[List[IUploadModelResponse]]:
        """
        Upload a model to Runware.

        Args:
            requestModel: Model upload request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            List of upload model responses
        """
        await self.ensureConnection()

        operation_timeout = timeout or self.default_timeout

        operation = ModelUploadOperation(requestModel, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def modelSearch(
        self,
        payload: IModelSearch,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> IModelSearchResponse:
        """
        Search for models in the Runware model library.

        Args:
            payload: Model search request parameters
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback function

        Returns:
            Model search response with results
        """
        await self.ensureConnection()

        operation_timeout = timeout or self.default_timeout

        operation = ModelSearchOperation(payload, self)

        if progress_callback:
            operation.add_progress_callback(progress_callback)

        return await self._execute_operation(operation, operation_timeout)

    async def _check_auto_disconnect(self):
        """Check if we should auto-disconnect after operation completion"""
        if not self.auto_disconnect or not self._is_started:
            return

        # Check if there are any active operations
        active_operations = len(self.operation_manager.operations)
        if active_operations == 0:
            # Schedule disconnect to run after current operation completes
            asyncio.create_task(self._delayed_disconnect())

    async def _delayed_disconnect(self):
        """Disconnect after a short delay to ensure operation cleanup"""
        try:
            await asyncio.sleep(0.1)  # Small delay to ensure cleanup
            if len(self.operation_manager.operations) == 0:
                await self.disconnect()
        except Exception as e:
            self.logger.debug(f"Error in delayed disconnect: {e}")

    # Operation management methods
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific operation"""
        context = self.operation_manager.get_operation_context(operation_id)
        if context:
            return {
                "operation_id": context.operation_id,
                "operation_type": context.operation_type,
                "status": context.status.value,
                "progress": context.progress,
                "created_at": context.created_at,
                "completed_at": context.completed_at,
                "results_count": len(context.results) if context.results else 0,
            }
        return None

    def list_operations(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        """List all current operations"""
        contexts = self.operation_manager.list_operations()

        operations = []
        for context in contexts:
            if not include_completed and context.status.value in [
                "completed",
                "failed",
                "cancelled",
            ]:
                continue

            operations.append(
                {
                    "operation_id": context.operation_id,
                    "operation_type": context.operation_type,
                    "status": context.status.value,
                    "progress": context.progress,
                    "created_at": context.created_at,
                    "completed_at": context.completed_at,
                }
            )

        return operations

    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a specific operation"""
        return await self.operation_manager.cancel_operation(operation_id)

    async def cancel_all_operations(self) -> int:
        """Cancel all active operations"""
        return await self.operation_manager.cancel_all_operations()

    async def wait_for_connection(self, timeout: Optional[float] = 30.0) -> bool:
        """Wait for connection to be established"""
        if not self._is_started:
            return False
        return await self.connection_manager.wait_for_authentication(timeout)

    def _on_connection_state_change(self, new_state: ConnectionState):
        """Handle connection state changes"""
        for callback in self._connection_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                self.logger.error("Error in connection callback", exc_info=e)

    async def _cleanup(self):
        """Cleanup all resources"""
        try:
            if self._is_started:
                await self.operation_manager.stop()
                await self.message_router.stop()
                await self.connection_manager.stop()
        except Exception as e:
            self.logger.error("Error during cleanup", exc_info=e)


# Backward compatibility alias
Runware = RunwareClient
RunwareServer = RunwareClient  # lol
