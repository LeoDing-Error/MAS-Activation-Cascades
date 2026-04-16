"""Model backends for steered and clean agent execution."""

from .camel_integration import (
    AgentSpec,
    create_chat_agent,
    create_clean_chat_agent,
    create_openai_compatible_agent,
    create_role_playing_session,
)
from .steering_backend import (
    CleanModelBackend,
    SteeringHook,
    SteeringModelBackend,
    SteeringVectorArtifact,
)

__all__ = [
    "AgentSpec",
    "CleanModelBackend",
    "SteeringHook",
    "SteeringModelBackend",
    "SteeringVectorArtifact",
    "create_chat_agent",
    "create_clean_chat_agent",
    "create_openai_compatible_agent",
    "create_role_playing_session",
]
