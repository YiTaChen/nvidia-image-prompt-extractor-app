from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT


COMFYUI_WORKFLOW_ROOT = PROJECT_ROOT / "backend" / "app" / "workflows" / "comfyui"
COMFYUI_WORKFLOW_CATALOG = COMFYUI_WORKFLOW_ROOT / "workflow_catalog.json"


@dataclass(frozen=True)
class ComfyUIWorkflowDefinition:
    id: str
    display_name: str
    mode: str
    workflow_path: Path
    bindings: dict[str, str]
    capabilities: list[str]
    description: str = ""
    required_checkpoint: str = ""
    required_custom_nodes: list[str] | None = None
    defaults: dict[str, Any] | None = None
    primary: bool = False


def list_workflow_definitions() -> list[ComfyUIWorkflowDefinition]:
    catalog = json.loads(COMFYUI_WORKFLOW_CATALOG.read_text(encoding="utf-8"))
    definitions = []
    for index, item in enumerate(catalog.get("workflows", [])):
        definitions.append(
            ComfyUIWorkflowDefinition(
                id=item["id"],
                display_name=item.get("display_name", item["id"]),
                mode=item["mode"],
                workflow_path=PROJECT_ROOT / item["workflow_path"],
                bindings=dict(item.get("bindings", {})),
                capabilities=list(item.get("capabilities", [])),
                description=item.get("description", ""),
                required_checkpoint=item.get("required_checkpoint", ""),
                required_custom_nodes=list(item.get("required_custom_nodes", [])),
                defaults=dict(item.get("defaults", {})),
                primary=index == 0,
            )
        )
    return definitions


def get_workflow_definition(workflow_id: str) -> ComfyUIWorkflowDefinition:
    for definition in list_workflow_definitions():
        if definition.id == workflow_id:
            return definition
    raise ValueError(f"Unknown ComfyUI workflow: {workflow_id}")


def load_workflow_prompt(workflow_id: str) -> dict[str, Any]:
    definition = get_workflow_definition(workflow_id)
    return json.loads(definition.workflow_path.read_text(encoding="utf-8"))


def patch_workflow(workflow_id: str, values: dict[str, Any]) -> dict[str, Any]:
    definition = get_workflow_definition(workflow_id)
    workflow = deepcopy(load_workflow_prompt(workflow_id))
    for binding_name, value in values.items():
        if value is None:
            continue
        binding_path = definition.bindings.get(binding_name)
        if not binding_path:
            continue
        _set_binding_value(workflow, binding_path, value)
    return workflow


def _set_binding_value(workflow: dict[str, Any], binding_path: str, value: Any) -> None:
    parts = binding_path.split(".")
    if len(parts) < 3:
        raise ValueError(f"Invalid workflow binding path: {binding_path}")

    current: Any = workflow
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"Workflow binding path not found: {binding_path}")

    final_key = parts[-1]
    if not isinstance(current, dict) or final_key not in current:
        raise ValueError(f"Workflow binding path not found: {binding_path}")
    current[final_key] = value
