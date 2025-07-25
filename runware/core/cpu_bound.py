import asyncio
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, TypeVar

T = TypeVar("T")


class CPUBoundExecutor:
    """Manages CPU-bound operations in thread pool."""

    def __init__(self, max_workers: int = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop = None

    async def __aenter__(self):
        self._loop = asyncio.get_event_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=True)

    async def parse_json(self, data: str) -> Dict[str, Any]:
        """Parse JSON in thread pool."""
        return await self._run_in_executor(json.loads, data)

    async def serialize_json(self, obj: Any) -> str:
        """Serialize to JSON in thread pool with enum support."""

        def json_dumps_with_enums(obj):
            def convert_enums(data):
                if hasattr(data, "value") and hasattr(data, "name"):  # It's an enum
                    return data.value
                elif isinstance(data, dict):
                    return {key: convert_enums(value) for key, value in data.items()}
                elif isinstance(data, list):
                    return [convert_enums(item) for item in data]
                else:
                    return data

            converted_obj = convert_enums(obj)
            return json.dumps(converted_obj)

        return await self._run_in_executor(json_dumps_with_enums, obj)

    async def serialize_dataclass(self, obj: Any) -> Dict[str, Any]:
        """Serialize dataclass in thread pool with enum support."""
        if is_dataclass(obj):

            def serialize_with_enums(obj):
                result = asdict(obj)
                # Convert enums to their string values
                for key, value in result.items():
                    if hasattr(value, "value") and hasattr(
                        value, "name"
                    ):  # It's an enum
                        result[key] = value.value
                return result

            return await self._run_in_executor(serialize_with_enums, obj)
        return obj

    async def batch_serialize_dataclasses(
        self, objects: List[Any]
    ) -> List[Dict[str, Any]]:
        """Serialize multiple dataclasses concurrently."""
        tasks = [self.serialize_dataclass(obj) for obj in objects]
        return await asyncio.gather(*tasks)

    async def _run_in_executor(self, func: Callable, *args, **kwargs):
        """Run function in thread pool."""
        loop = self._loop or asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(func, *args, **kwargs)
        )


# Global instance
cpu_executor = CPUBoundExecutor(max_workers=4)
