# pta_reload/__init__.py
__version__ = "0.1.0"

class _Imports:
    """Configuration for external library imports"""
    def __init__(self):
        self._imports = {
            "talib": False
        }
    
    def __getitem__(self, key):
        return self._imports.get(key, False)
    
    def __setitem__(self, key, value):
        self._imports[key] = value

# Create INSTANCE instead of using class directly
Imports = _Imports()

from .ta import ta  # Import after defining Imports
__all__ = ["ta", "Imports"]
