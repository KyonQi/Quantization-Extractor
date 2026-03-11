from .base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}

def register(name: str):
    def decorator(cls: type[ModelAdapter]):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_adapter(name: str) -> ModelAdapter:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found in registry. Available models: {available}")
    return _REGISTRY[name]()

def list_models() -> list[str]:
    return list(_REGISTRY.keys())