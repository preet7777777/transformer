"""Simple registry helper."""

from __future__ import annotations

from typing import Any, ClassVar

_REGISTRIES: dict[type, dict[str, type]] = {}


class Registrable:
    """Register subclasses under string names."""

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            _REGISTRIES.setdefault(cls, {})[name] = subclass
            return subclass

        return decorator

    @classmethod
    def by_name(cls, name: str):
        registry = _REGISTRIES.get(cls, {})
        if name not in registry:
            raise KeyError(f"{name!r} is not registered for {cls.__name__}")
        return registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(_REGISTRIES.get(cls, {}).keys())
