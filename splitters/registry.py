"""The splitter registry: select and introspect methods by name.

``@register("name")`` (in :mod:`.base`) populates ``_REGISTRY``. The built-in
methods register themselves when :mod:`PALM.splitters.methods` is imported.
"""

from __future__ import annotations

from typing import Any, Dict, List

# name -> BaseSplitter subclass. Populated by the @register decorator.
_REGISTRY: Dict[str, type] = {}


def get_splitter(name: str, **params):
    """Instantiate the splitter registered under ``name`` with ``params``.

    Raises ``KeyError`` (with the list of known names) if ``name`` is unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown splitter '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def list_splitters() -> List[str]:
    """Sorted names of all registered splitters."""
    return sorted(_REGISTRY)


def describe_splitters() -> List[Dict[str, Any]]:
    """``[{name, description, arity, params_schema}, ...]`` for every splitter.

    This is the discovery payload an agent / MCP tool reads to decide which
    method to call and with what parameters.
    """
    return [_REGISTRY[n].describe() for n in list_splitters()]
