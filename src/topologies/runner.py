from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.project_paths import ensure_local_camel_on_path

ensure_local_camel_on_path()

from camel.agents import ChatAgent
from camel.responses import ChatAgentResponse
from camel.societies.workforce import Workforce
from camel.societies.workforce.events import (
    AllTasksCompletedEvent,
    LogEvent,
    TaskAssignedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskDecomposedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    TaskUpdatedEvent,
    WorkerCreatedEvent,
    WorkerDeletedEvent,
)
from camel.societies.workforce.workforce_callback import WorkforceCallback

from src.backends.camel_integration import create_role_playing_session
from src.metrics.uncertainty import CascadeUncertaintyTracker, snapshot_from_backend


@dataclass
class AgentNode:
    agent_id: str
    role_name: str
    hop: int
    agent: ChatAgent


@dataclass
class TopologyRunResult:
    topology: str
    condition: str
    task_prompt: str
    messages: List[Dict[str, Any]]
    uncertainty: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
            "condition": self.condition,
            "task_prompt": self.task_prompt,
            "messages": self.messages,
            "uncertainty": self.uncertainty,
            "metadata": self.metadata,
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class WorkforceEventRecorder(WorkforceCallback):
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def _record(self, event: Any) -> None:
        payload = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        self.events.append(payload)

    def log_message(self, event: LogEvent) -> None:
        self._record(event)

    def log_task_created(self, event: TaskCreatedEvent) -> None:
        self._record(event)

    def log_task_decomposed(self, event: TaskDecomposedEvent) -> None:
        self._record(event)

    def log_task_assigned(self, event: TaskAssignedEvent) -> None:
        self._record(event)

    def log_task_started(self, event: TaskStartedEvent) -> None:
        self._record(event)

    def log_task_updated(self, event: TaskUpdatedEvent) -> None:
        self._record(event)

    def log_task_completed(self, event: TaskCompletedEvent) -> None:
        self._record(event)

    def log_task_failed(self, event: TaskFailedEvent) -> None:
        self._record(event)

    def log_worker_created(self, event: WorkerCreatedEvent) -> None:
        self._record(event)

    def log_worker_deleted(self, event: WorkerDeletedEvent) -> None:
        self._record(event)

    def log_all_tasks_completed(self, event: AllTasksCompletedEvent) -> None:
        self._record(event)


class CascadeTopologyRunner:
    def __init__(self, tracker: Optional[CascadeUncertaintyTracker] = None) -> None:
        self.tracker = tracker or CascadeUncertaintyTracker()

    def run_single_agent(
        self,
        *,
        agent: AgentNode,
        task_prompt: str,
        condition: str,
    ) -> TopologyRunResult:
        response = agent.agent.step(task_prompt)
        text = self._response_text(response)
        messages = [
            self._log_message(
                agent=agent,
                topology="single",
                condition=condition,
                turn=0,
                speaker="assistant",
                content=text,
                metadata={"terminated": response.terminated},
            )
        ]
        return TopologyRunResult(
            topology="single",
            condition=condition,
            task_prompt=task_prompt,
            messages=messages,
            uncertainty=self.tracker.to_list(),
            metadata={},
        )

    def run_two_agent_chain(
        self,
        *,
        source: AgentNode,
        target: AgentNode,
        task_prompt: str,
        condition: str,
        chat_turn_limit: int = 3,
    ) -> TopologyRunResult:
        messages, relay, _ = self._run_role_play_pair(
            source=source,
            target=target,
            task_prompt=task_prompt,
            topology="chain_two",
            condition=condition,
            chat_turn_limit=chat_turn_limit,
        )
        return TopologyRunResult(
            topology="chain_two",
            condition=condition,
            task_prompt=task_prompt,
            messages=messages,
            uncertainty=self.tracker.to_list(),
            metadata={"final_relay": relay},
        )

    def run_three_agent_chain(
        self,
        *,
        source: AgentNode,
        middle: AgentNode,
        target: AgentNode,
        task_prompt: str,
        condition: str,
        chat_turn_limit: int = 2,
    ) -> TopologyRunResult:
        pair_one, relay, next_turn = self._run_role_play_pair(
            source=source,
            target=middle,
            task_prompt=task_prompt,
            topology="chain_three",
            condition=condition,
            chat_turn_limit=chat_turn_limit,
        )
        pair_two_prompt = (
            f"Original task:\n{task_prompt}\n\n"
            f"Upstream context from {middle.role_name}:\n{relay}"
        )
        pair_two, final_relay, _ = self._run_role_play_pair(
            source=middle,
            target=target,
            task_prompt=pair_two_prompt,
            topology="chain_three",
            condition=condition,
            chat_turn_limit=chat_turn_limit,
            turn_offset=next_turn,
        )
        return TopologyRunResult(
            topology="chain_three",
            condition=condition,
            task_prompt=task_prompt,
            messages=pair_one + pair_two,
            uncertainty=self.tracker.to_list(),
            metadata={"midpoint_relay": relay, "final_relay": final_relay},
        )

    def run_star_topology(
        self,
        *,
        hub: AgentNode,
        peripherals: Sequence[AgentNode],
        task_prompt: str,
        condition: str,
    ) -> TopologyRunResult:
        event_recorder = WorkforceEventRecorder()
        workforce = Workforce(
            description="Cascade star topology",
            callbacks=[event_recorder],
            use_structured_output_handler=True,
        )
        for peripheral in peripherals:
            workforce.add_single_agent_worker(description=peripheral.role_name, worker=peripheral.agent)

        messages: List[Dict[str, Any]] = []
        hub_response = hub.agent.step(task_prompt)
        hub_text = self._response_text(hub_response)
        messages.append(
            self._log_message(
                agent=hub,
                topology="star",
                condition=condition,
                turn=0,
                speaker="hub",
                content=hub_text,
                metadata={"terminated": hub_response.terminated},
            )
        )

        # We use Workforce to define and log the star roster, but broadcast is explicit so every leaf sees the same upstream text.
        for peripheral in peripherals:
            peripheral_prompt = (
                f"Hub message from {hub.role_name}:\n{hub_text}\n\n"
                f"You are acting as the {peripheral.role_name}. Respond to the task below using the hub context above.\n\n"
                f"Task:\n{task_prompt}"
            )
            response = peripheral.agent.step(peripheral_prompt)
            text = self._response_text(response)
            messages.append(
                self._log_message(
                    agent=peripheral,
                    topology="star",
                    condition=condition,
                    turn=1,
                    speaker="leaf",
                    content=text,
                    metadata={"terminated": response.terminated},
                )
            )

        return TopologyRunResult(
            topology="star",
            condition=condition,
            task_prompt=task_prompt,
            messages=messages,
            uncertainty=self.tracker.to_list(),
            metadata={"workforce_events": event_recorder.events},
        )

    def _run_role_play_pair(
        self,
        *,
        source: AgentNode,
        target: AgentNode,
        task_prompt: str,
        topology: str,
        condition: str,
        chat_turn_limit: int,
        turn_offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], str, int]:
        session = create_role_playing_session(
            source_agent=source.agent,
            target_agent=target.agent,
            source_role_name=source.role_name,
            target_role_name=target.role_name,
            task_prompt=task_prompt,
            with_task_specify=False,
        )
        messages: List[Dict[str, Any]] = []
        relay_text = task_prompt
        input_msg = session.init_chat()
        turn = turn_offset

        messages.append(
            self._log_message(
                agent=source,
                topology=topology,
                condition=condition,
                turn=turn,
                speaker="source_init",
                content=input_msg.content,
                metadata={"stage": "init_chat"},
            )
        )

        while turn < chat_turn_limit:
            assistant_response, user_response = session.step(input_msg)
            target_text = self._response_text(user_response)
            relay_text = target_text or relay_text
            if target_text:
                messages.append(
                    self._log_message(
                        agent=target,
                        topology=topology,
                        condition=condition,
                        turn=turn,
                        speaker="target",
                        content=target_text,
                        metadata={"terminated": user_response.terminated},
                    )
                )
            if user_response.terminated or "CAMEL_TASK_DONE" in target_text:
                break

            source_text = self._response_text(assistant_response)
            if not source_text:
                break
            turn += 1
            messages.append(
                self._log_message(
                    agent=source,
                    topology=topology,
                    condition=condition,
                    turn=turn,
                    speaker="source",
                    content=source_text,
                    metadata={"terminated": assistant_response.terminated},
                )
            )
            if assistant_response.terminated or assistant_response.msg is None:
                break
            input_msg = assistant_response.msg
        return messages, relay_text, turn + 1

    def _log_message(
        self,
        *,
        agent: AgentNode,
        topology: str,
        condition: str,
        turn: int,
        speaker: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metrics = snapshot_from_backend(agent.agent.model_backend, text=content)
        self.tracker.record(
            agent_id=agent.agent_id,
            role_name=agent.role_name,
            topology=topology,
            condition=condition,
            hop=agent.hop,
            turn=turn,
            text=content,
            metrics=metrics,
            metadata=metadata,
        )
        return {
            "agent_id": agent.agent_id,
            "role_name": agent.role_name,
            "hop": agent.hop,
            "turn": turn,
            "speaker": speaker,
            "content": content,
            "metadata": metadata or {},
        }

    def _response_text(self, response: ChatAgentResponse) -> str:
        if response.msg is not None:
            return response.msg.content
        if response.msgs:
            return response.msgs[0].content
        return ""
