from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.project_paths import ensure_local_camel_on_path

ensure_local_camel_on_path()

from camel.agents import ChatAgent
from camel.models import OpenAICompatibleModel
from camel.societies import RolePlaying

from .steering_backend import CleanModelBackend, SteeringModelBackend

DEFAULT_SYSTEM_MESSAGE = "You are a careful collaborative coding agent. Keep your replies concise and task-focused."


@dataclass
class AgentSpec:
    agent_id: str
    role_name: str
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)


def create_chat_agent(
    spec: AgentSpec,
    *,
    model_name: str,
    steering_vector_path: str,
    steering_strength: float = 1.0,
    steering_enabled: bool = True,
) -> ChatAgent:
    backend = SteeringModelBackend(
        model_name=model_name,
        steering_path=steering_vector_path,
        alpha=steering_strength,
        steering_enabled=steering_enabled,
        **spec.backend_kwargs,
    )
    return ChatAgent(system_message=spec.system_message, model=backend, **spec.agent_kwargs)


def create_clean_chat_agent(
    spec: AgentSpec,
    *,
    model_name: str,
) -> ChatAgent:
    backend = CleanModelBackend(model_name=model_name, **spec.backend_kwargs)
    return ChatAgent(system_message=spec.system_message, model=backend, **spec.agent_kwargs)


def create_openai_compatible_agent(
    spec: AgentSpec,
    *,
    model_name: str,
    api_base_url: str,
    api_key: str = "EMPTY",
    model_config_dict: Optional[Dict[str, Any]] = None,
) -> ChatAgent:
    backend = OpenAICompatibleModel(
        model_type=model_name,
        model_config_dict=model_config_dict or {},
        api_key=api_key,
        url=api_base_url,
    )
    return ChatAgent(system_message=spec.system_message, model=backend, **spec.agent_kwargs)


def create_role_playing_session(
    *,
    source_agent: ChatAgent,
    target_agent: ChatAgent,
    source_role_name: str,
    target_role_name: str,
    task_prompt: str,
    with_task_specify: bool = False,
) -> RolePlaying:
    return RolePlaying(
        assistant_role_name=source_role_name,
        user_role_name=target_role_name,
        assistant_agent=source_agent,
        user_agent=target_agent,
        task_prompt=task_prompt,
        with_task_specify=with_task_specify,
    )
