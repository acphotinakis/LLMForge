# Re-export from seed.py so utils/__init__.py imports cleanly
from .seed import get_device, get_dtype

__all__ = ["get_device", "get_dtype"]
