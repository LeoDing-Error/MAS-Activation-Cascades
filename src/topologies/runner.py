from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.project_paths import ensure_local_camel_on_path

ensure_local_camel_on_path()

from camel.agents import ChatAgent
from camel.messages.base import BaseMessage
from camel.responses import ChatAgentResponse
from camel.societies.workforce import Workforce
from camel.societies.workforce import WorkforceMode
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
from camel.tasks.task import Task

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
        workforce = self._build_star_workforce(
            hub=hub,
            peripherals=peripherals,
            event_recorder=event_recorder,
        )
        hub_task, leaf_tasks = self._build_star_pipeline_tasks(
            task_prompt=task_prompt,
            hub=hub,
            peripherals=peripherals,
            condition=condition,
        )
        workforce.set_pipeline_tasks([hub_task, *leaf_tasks])

        root_task = Task(
            content=task_prompt,
            id=f"{condition}_star_root",
            additional_info={
                "topology": "star",
                "condition": condition,
                "mode": "workforce_pipeline",
            },
        )
        root_task = workforce.process_task(root_task)
        worker_lookup = {
            agent.agent_id: agent for agent in [hub, *peripherals]
        }

        messages: List[Dict[str, Any]] = []
        assigned_hub = self._resolve_assigned_agent(
            assigned_worker_id=hub_task.assigned_worker_id,
            worker_lookup=worker_lookup,
            fallback=hub,
        )
        messages.append(
            self._log_message(
                agent=assigned_hub,
                topology="star",
                condition=condition,
                turn=0,
                speaker="hub",
                content=hub_task.result or "",
                metadata={
                    "task_id": hub_task.id,
                    "assigned_worker_id": hub_task.assigned_worker_id,
                    "task_state": hub_task.state.value,
                    "dependencies": [
                        dep.id for dep in hub_task.dependencies
                    ]
                    if hub_task.dependencies
                    else [],
                    "terminated": hub_task.state.value == "DONE",
                },
            )
        )
        for peripheral, leaf_task in zip(peripherals, leaf_tasks):
            assigned_leaf = self._resolve_assigned_agent(
                assigned_worker_id=leaf_task.assigned_worker_id,
                worker_lookup=worker_lookup,
                fallback=peripheral,
            )
            messages.append(
                self._log_message(
                    agent=assigned_leaf,
                    topology="star",
                    condition=condition,
                    turn=1,
                    speaker="leaf",
                    content=leaf_task.result or "",
                    metadata={
                        "task_id": leaf_task.id,
                        "assigned_worker_id": leaf_task.assigned_worker_id,
                        "task_state": leaf_task.state.value,
                        "dependencies": [
                            dep.id for dep in leaf_task.dependencies
                        ]
                        if leaf_task.dependencies
                        else [],
                        "terminated": leaf_task.state.value == "DONE",
                    },
                )
            )

        pipeline_task_ids = [hub_task.id, *[task.id for task in leaf_tasks]]

        return TopologyRunResult(
            topology="star",
            condition=condition,
            task_prompt=task_prompt,
            messages=messages,
            uncertainty=self.tracker.to_list(),
            metadata={
                "workforce_events": event_recorder.events,
                "pipeline_task_ids": pipeline_task_ids,
                "root_task_state": root_task.state.value,
                "root_task_result": root_task.result,
            },
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
        metrics = snapshot_from_backend(agent.agent, text=content)
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

    @staticmethod
    def _resolve_assigned_agent(
        *,
        assigned_worker_id: Optional[str],
        worker_lookup: Dict[str, AgentNode],
        fallback: AgentNode,
    ) -> AgentNode:
        if assigned_worker_id is None:
            return fallback
        return worker_lookup.get(assigned_worker_id, fallback)

    def _response_text(self, response: ChatAgentResponse) -> str:
        if response.msg is not None:
            return response.msg.content
        if response.msgs:
            return response.msgs[0].content
        return ""

    def _build_star_workforce(
        self,
        *,
        hub: AgentNode,
        peripherals: Sequence[AgentNode],
        event_recorder: WorkforceEventRecorder,
    ) -> Workforce:
        if not peripherals:
            raise ValueError("Star topology requires at least one peripheral")

        orchestration_template = peripherals[0].agent.clone(with_memory=False)
        coordinator_agent = orchestration_template.clone(with_memory=False)
        coordinator_agent.update_system_message(
            BaseMessage.make_assistant_message(
                role_name="Workforce Coordinator",
                content=(
                    "You coordinate star-topology tasks. Assign the hub task "
                    "to the hub worker and leaf tasks to the matching "
                    "specialist workers. Keep orchestration separate from "
                    "the task content itself."
                ),
            )
        )
        task_agent = orchestration_template.clone(with_memory=False)
        task_agent.update_system_message(
            BaseMessage.make_assistant_message(
                role_name="Task Planner",
                content=(
                    "You plan star-topology pipeline tasks. Keep the hub "
                    "broadcast and leaf follow-up tasks aligned with the "
                    "original prompt."
                ),
            )
        )

        workforce = Workforce(
            description="Cascade star topology",
            coordinator_agent=coordinator_agent,
            task_agent=task_agent,
            callbacks=[event_recorder],
            use_structured_output_handler=True,
            mode=WorkforceMode.PIPELINE,
        )
        workforce.add_single_agent_worker(
            description=self._star_worker_description(
                role_name=hub.role_name,
                kind="hub",
            ),
            worker=hub.agent,
        )
        for peripheral in peripherals:
            workforce.add_single_agent_worker(
                description=self._star_worker_description(
                    role_name=peripheral.role_name,
                    kind="leaf",
                ),
                worker=peripheral.agent,
            )
        return workforce

    def _build_star_pipeline_tasks(
        self,
        *,
        task_prompt: str,
        hub: AgentNode,
        peripherals: Sequence[AgentNode],
        condition: str,
    ) -> tuple[Task, List[Task]]:
        hub_task = Task(
            content=(
                f"Star hub task for the {hub.role_name} worker.\n\n"
                f"Original task:\n{task_prompt}\n\n"
                "Produce a concise broadcast that summarizes the task for "
                "the specialist leaf workers."
            ),
            id=f"{condition}_star_hub",
            additional_info={
                "topology": "star",
                "condition": condition,
                "stage": "hub",
                "role_name": hub.role_name,
                "leaf_roles": [peripheral.role_name for peripheral in peripherals],
            },
        )

        leaf_tasks: List[Task] = []
        for peripheral in peripherals:
            leaf_task = Task(
                content=(
                    f"Star leaf task for the {peripheral.role_name} worker.\n\n"
                    f"Original task:\n{task_prompt}\n\n"
                    "Use the hub broadcast from the dependency info and "
                    "return a role-specific response for your specialty."
                ),
                id=f"{condition}_star_{peripheral.role_name}",
                additional_info={
                    "topology": "star",
                    "condition": condition,
                    "stage": "leaf",
                    "role_name": peripheral.role_name,
                    "hub_role": hub.role_name,
                },
            )
            leaf_task.dependencies = [hub_task]
            leaf_tasks.append(leaf_task)

        return hub_task, leaf_tasks

    @staticmethod
    def _star_worker_description(*, role_name: str, kind: str) -> str:
        if kind == "hub":
            return (
                f"Central star hub worker for {role_name}. "
                "Consolidates the task into a broadcast for the leaves."
            )
        return (
            f"Star leaf specialist for {role_name}. "
            "Consumes the hub broadcast and responds with a role-specific "
            "specialization."
        )
