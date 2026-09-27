"""Model registry.

Reads ``model_config.yaml`` and hands out a model handle per task. Today every
handle is a :class:`MockModel` that echoes a plausible payload, so the API
surface can be exercised end to end. Swapping in a real specialist node means
registering a loader for that task name — nothing else moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

TaskName = str
ModelLoader = Callable[[TaskName, dict[str, Any]], Any]


class UnknownTaskError(KeyError):
    """Raised when a task has no entry in model_config.yaml."""


@dataclass
class MockModel:
    """Placeholder standing in for a not-yet-wired specialist model node."""

    task: TaskName
    config: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False

    def predict(self, **inputs: Any) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": "mock",
            "config": self.config,
            "inputs": sorted(inputs),
            "result": f"[mock] {self.task} not yet wired to a real checkpoint",
            "confidence": 0.0,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug affordance
        return f"MockModel(task={self.task!r}, loaded={self.loaded})"


class ModelRegistry:
    """Lazily resolves task names to model handles, caching each one."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._config: dict[str, dict[str, Any]] = self._read_config()
        self._loaders: dict[TaskName, ModelLoader] = {}
        self._cache: dict[TaskName, Any] = {}

    def _read_config(self) -> dict[str, dict[str, Any]]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"model config not found: {self.config_path}")
        with self.config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self.config_path} must be a mapping of task -> config")
        return {task: (cfg or {}) for task, cfg in raw.items()}

    @property
    def tasks(self) -> list[TaskName]:
        return sorted(self._config)

    def config_for(self, task_name: TaskName) -> dict[str, Any]:
        try:
            return dict(self._config[task_name])
        except KeyError as exc:
            raise UnknownTaskError(
                f"unknown task {task_name!r}; known tasks: {', '.join(self.tasks)}"
            ) from exc

    def register_loader(self, task_name: TaskName, loader: ModelLoader) -> None:
        """Attach a real loader for a task. Drops the cached handle, if any."""
        self._loaders[task_name] = loader
        self._cache.pop(task_name, None)

    def get_model(self, task_name: TaskName) -> Any:
        if task_name in self._cache:
            return self._cache[task_name]

        config = self.config_for(task_name)
        # The CPU fallback is declared once, under vqa_grounding, and applies
        # to every engine that wraps the VQA model.
        if task_name != "vqa_grounding" and "cpu_fallback" not in config:
            config["cpu_fallback"] = self._config.get("vqa_grounding", {}).get("cpu_fallback")
        loader = self._loaders.get(task_name)
        if loader is None:
            logger.info("no loader registered for %r — returning mock", task_name)
            model = MockModel(task=task_name, config=config)
        else:
            model = loader(task_name, config)

        self._cache[task_name] = model
        return model

    def reload(self) -> None:
        """Re-read the YAML and drop every cached handle."""
        self._config = self._read_config()
        self._cache.clear()


def _load_geochat(task_name: TaskName, config: dict[str, Any]) -> Any:
    """Build the VQA engine for `vqa_grounding`.

    The checkpoint comes from `model_config.yaml` — stock LLaVA-1.5 today; the
    engine also carries a loader path for GeoChat-format weights. Constructing
    it is cheap — no weights are touched until the first `infer` call — so a
    machine that can never load the checkpoint still starts, and fails with a
    readable reason only when someone actually asks a question.
    """
    from app.models.geochat import GeoChatConfig, GeoChatEngine

    return GeoChatEngine(GeoChatConfig.from_mapping(config))


def _load_change(task_name: TaskName, config: dict[str, Any]) -> Any:
    """Build the prompted-diff change detector for `change_detection`.

    Shares the VQA checkpoint — the vision tower locates the change and the
    language head describes it — so this costs no second set of weights.
    """
    from app.models.change import ChangeConfig, ChangeDetectorEngine

    return ChangeDetectorEngine(ChangeConfig.from_mapping(config))


def _load_fusion(task_name: TaskName, config: dict[str, Any]) -> Any:
    """Build the optical-SAR fusion engine for `optical_sar_fusion`.

    Constructing it never touches the encoder weights, so a machine without
    SSL4EO-S12 or DeCUR checkpoints still starts and reports the task
    unavailable only when someone asks a fusion question.
    """
    from app.models.fusion import FusionConfig, FusionEngine

    return FusionEngine(FusionConfig.from_mapping(config))


#: Loaders for tasks that have a real implementation. Anything absent here
#: resolves to a MockModel, which is how a further specialist gets dropped in.
DEFAULT_LOADERS: dict[TaskName, ModelLoader] = {
    "vqa_grounding": _load_geochat,
    "change_detection": _load_change,
    "optical_sar_fusion": _load_fusion,
}


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """FastAPI dependency — one process-wide registry."""
    global _registry
    if _registry is None:
        from app.core.config import get_settings

        _registry = ModelRegistry(get_settings().model_config_path)
        for task, loader in DEFAULT_LOADERS.items():
            if task in _registry.tasks:
                _registry.register_loader(task, loader)
    return _registry
