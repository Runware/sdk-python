import logging
import sys
from typing import Any, Dict


def setup_logging(
    log_level: str = "INFO", detailed_websockets: bool = False
) -> logging.Logger:
    """
    Setup comprehensive logging for the entire Runware SDK with proper string level support.

    Args:
        log_level: String log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        detailed_websockets: Whether to enable detailed websocket message logging

    Returns:
        The main SDK logger instance
    """
    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(
                f"Invalid log level: {log_level}. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
    else:
        numeric_level = log_level

    # Create comprehensive formatter with more detailed format
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create simple formatter for less verbose components
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger for the SDK
    sdk_logger = logging.getLogger("runware")
    sdk_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in sdk_logger.handlers[:]:
        sdk_logger.removeHandler(handler)

    # Add console handler for main SDK
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel(numeric_level)
    sdk_logger.addHandler(console_handler)

    # Setup websockets logger with configurable detail level
    websockets_logger = logging.getLogger("websockets")

    # Remove existing handlers
    for handler in websockets_logger.handlers[:]:
        websockets_logger.removeHandler(handler)

    if detailed_websockets:
        # Detailed websocket logging when requested
        websockets_logger.setLevel(max(numeric_level, logging.DEBUG))
        websockets_logger.propagate = False

        websocket_handler = logging.StreamHandler(sys.stdout)
        websocket_handler.setFormatter(detailed_formatter)
        websocket_handler.setLevel(max(numeric_level, logging.DEBUG))
        websockets_logger.addHandler(websocket_handler)
    else:
        # Standard websocket logging
        websockets_logger.setLevel(max(numeric_level, logging.INFO))
        websockets_logger.propagate = False

        websocket_handler = logging.StreamHandler(sys.stdout)
        websocket_handler.setFormatter(simple_formatter)
        websocket_handler.setLevel(max(numeric_level, logging.INFO))
        websockets_logger.addHandler(websocket_handler)

    # Setup asyncio logger with appropriate level
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.setLevel(
        max(numeric_level, logging.WARNING)
    )  # Only warnings and errors for asyncio

    # Configure specific component loggers with appropriate levels
    component_configs = {
        "runware.client": numeric_level,
        "runware.operations": numeric_level,
        "runware.connection": numeric_level,
        "runware.messaging": numeric_level,
        "runware.core": numeric_level,
    }

    for component_name, level in component_configs.items():
        component_logger = logging.getLogger(component_name)
        component_logger.setLevel(level)

    # Log the successful setup
    sdk_logger.info(f"Runware SDK logging initialized with level: {log_level}")
    if detailed_websockets:
        sdk_logger.debug("Detailed websocket logging enabled")

    return sdk_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module within the SDK.

    Args:
        name: The name of the module/component requesting the logger

    Returns:
        A configured logger instance for the specified component
    """
    if not name.startswith("runware"):
        if name == "__main__":
            logger_name = "runware.main"
        elif "." in name:
            module_parts = name.split(".")
            if len(module_parts) >= 2:
                logger_name = f"runware.{module_parts[-2]}.{module_parts[-1]}"
            else:
                logger_name = f"runware.{module_parts[-1]}"
        else:
            logger_name = f"runware.{name}"
    else:
        logger_name = name

    return logging.getLogger(logger_name)


def configure_component_logging(component_name: str, level: str) -> logging.Logger:
    """
    Configure logging for a specific component with a custom level.

    Args:
        component_name: Name of the component (e.g., 'websockets', 'connection')
        level: Log level for this component

    Returns:
        The configured logger for the component
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
    else:
        numeric_level = level

    logger = get_logger(component_name)
    logger.setLevel(numeric_level)

    return logger


def get_logging_stats() -> Dict[str, Any]:
    """
    Get statistics about the current logging configuration.

    Returns:
        Dictionary containing logging statistics and configuration info
    """
    runware_logger = logging.getLogger("runware")
    websockets_logger = logging.getLogger("websockets")
    asyncio_logger = logging.getLogger("asyncio")

    return {
        "main_level": logging.getLevelName(runware_logger.level),
        "websockets_level": logging.getLevelName(websockets_logger.level),
        "asyncio_level": logging.getLevelName(asyncio_logger.level),
        "handlers_count": {
            "runware": len(runware_logger.handlers),
            "websockets": len(websockets_logger.handlers),
            "asyncio": len(asyncio_logger.handlers),
        },
        "effective_levels": {
            "runware": logging.getLevelName(runware_logger.getEffectiveLevel()),
            "websockets": logging.getLevelName(websockets_logger.getEffectiveLevel()),
            "asyncio": logging.getLevelName(asyncio_logger.getEffectiveLevel()),
        },
    }

