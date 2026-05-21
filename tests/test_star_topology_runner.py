from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _EventBase:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class _FakeTask:
    def __init__(self, content: str, id: str, additional_info=None):
        self.content = content
        self.id = id
        self.additional_info = additional_info or {}
        self.dependencies = []
        self.assigned_worker_id = None
        self.result = None
        self.state = SimpleNamespace(value="PENDING")


class _FakeBaseMessage:
    @classmethod
    def make_assistant_message(cls, **kwargs):
        return kwargs


class _FakeWorkforceMode:
    PIPELINE = "PIPELINE"


class _FakeWorkforce:
    def __init__(self, *args, **kwargs):
        pass


class _FakeWorkforceCallback:
    pass


class _FakeChatAgentResponse:
    def __init__(self, msg=None, msgs=None, terminated=False):
        self.msg = msg
        self.msgs = msgs or []
        self.terminated = terminated


camel_module = types.ModuleType("camel")
camel_agents = types.ModuleType("camel.agents")
camel_agents.ChatAgent = object
camel_messages_base = types.ModuleType("camel.messages.base")
camel_messages_base.BaseMessage = _FakeBaseMessage
camel_responses = types.ModuleType("camel.responses")
camel_responses.ChatAgentResponse = _FakeChatAgentResponse
camel_workforce = types.ModuleType("camel.societies.workforce")
camel_workforce.Workforce = _FakeWorkforce
camel_workforce.WorkforceMode = _FakeWorkforceMode
camel_workforce_events = types.ModuleType("camel.societies.workforce.events")
for _name in [
    "AllTasksCompletedEvent",
    "LogEvent",
    "TaskAssignedEvent",
    "TaskCompletedEvent",
    "TaskCreatedEvent",
    "TaskDecomposedEvent",
    "TaskFailedEvent",
    "TaskStartedEvent",
    "TaskUpdatedEvent",
    "WorkerCreatedEvent",
    "WorkerDeletedEvent",
]:
    setattr(camel_workforce_events, _name, type(_name, (_EventBase,), {}))
camel_workforce_callback = types.ModuleType("camel.societies.workforce.workforce_callback")
camel_workforce_callback.WorkforceCallback = _FakeWorkforceCallback
camel_tasks_task = types.ModuleType("camel.tasks.task")
camel_tasks_task.Task = _FakeTask
fake_camel_integration = types.ModuleType("src.backends.camel_integration")
fake_camel_integration.create_role_playing_session = lambda *args, **kwargs: None

sys.modules.setdefault("camel", camel_module)
sys.modules["camel.agents"] = camel_agents
sys.modules["camel.messages.base"] = camel_messages_base
sys.modules["camel.responses"] = camel_responses
sys.modules["camel.societies.workforce"] = camel_workforce
sys.modules["camel.societies.workforce.events"] = camel_workforce_events
sys.modules["camel.societies.workforce.workforce_callback"] = camel_workforce_callback
sys.modules["camel.tasks.task"] = camel_tasks_task
sys.modules["src.backends.camel_integration"] = fake_camel_integration

# NOTE: torch is intentionally NOT stubbed here. Real torch is present in the
# cascade environment, and replacing sys.modules["torch"] with a fake module
# leaks across the whole pytest process, breaking tensor-math tests (e.g.
# test_uncertainty, test_compute_vectors) whose modules are imported afterwards.

import src.topologies.runner as runner_module
from src.topologies.runner import AgentNode, CascadeTopologyRunner


class FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id


class FakeRuntimeWorkforce:
    def __init__(self) -> None:
        self.pipeline_tasks = None

    def set_pipeline_tasks(self, tasks):
        self.pipeline_tasks = tasks

    def process_task(self, _root_task):
        return SimpleNamespace(
            state=SimpleNamespace(value="DONE"),
            result="root-complete",
        )


def _make_task(task_id: str, *, assigned_worker_id: str, result: str, dependencies=None):
    return SimpleNamespace(
        id=task_id,
        assigned_worker_id=assigned_worker_id,
        result=result,
        state=SimpleNamespace(value="DONE"),
        dependencies=list(dependencies or []),
    )


class StarTopologyRunnerTests(unittest.TestCase):
    def test_run_star_topology_attributes_leaf_outputs_to_assigned_worker(self) -> None:
        runner = CascadeTopologyRunner()

        hub = AgentNode(
            agent_id="hub-id",
            role_name="Hub",
            hop=0,
            agent=FakeAgent("hub-id"),
        )
        analyst = AgentNode(
            agent_id="analyst-id",
            role_name="Analyst",
            hop=1,
            agent=FakeAgent("analyst-id"),
        )
        reviewer = AgentNode(
            agent_id="reviewer-id",
            role_name="Reviewer",
            hop=1,
            agent=FakeAgent("reviewer-id"),
        )

        fake_workforce = FakeRuntimeWorkforce()
        hub_task = _make_task(
            "hub-task",
            assigned_worker_id="hub-id",
            result="hub broadcast",
        )
        analyst_task = _make_task(
            "analyst-task",
            assigned_worker_id="reviewer-id",
            result="response produced by reviewer",
            dependencies=[hub_task],
        )
        reviewer_task = _make_task(
            "reviewer-task",
            assigned_worker_id="analyst-id",
            result="response produced by analyst",
            dependencies=[hub_task],
        )

        with patch.object(
            runner,
            "_build_star_workforce",
            return_value=fake_workforce,
        ), patch.object(
            runner,
            "_build_star_pipeline_tasks",
            return_value=(hub_task, [analyst_task, reviewer_task]),
        ):
            result = runner.run_star_topology(
                hub=hub,
                peripherals=[analyst, reviewer],
                task_prompt="solve task",
                condition="baseline",
            )

        self.assertEqual(
            [message["agent_id"] for message in result.messages],
            ["hub-id", "reviewer-id", "analyst-id"],
        )
        self.assertEqual(
            [message["role_name"] for message in result.messages],
            ["Hub", "Reviewer", "Analyst"],
        )
        self.assertEqual(result.messages[1]["metadata"]["assigned_worker_id"], "reviewer-id")
        self.assertEqual(result.messages[2]["metadata"]["assigned_worker_id"], "analyst-id")

        self.assertEqual(
            [record["agent_id"] for record in result.uncertainty],
            ["hub-id", "reviewer-id", "analyst-id"],
        )
        self.assertEqual(
            [record["role_name"] for record in result.uncertainty],
            ["Hub", "Reviewer", "Analyst"],
        )
        self.assertEqual(
            [record["text"] for record in result.uncertainty],
            [
                "hub broadcast",
                "response produced by reviewer",
                "response produced by analyst",
            ],
        )


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeSession:
    """Scripted stand-in for a CAMEL role-playing session."""

    def __init__(self, init_content: str, steps) -> None:
        self._init_content = init_content
        self._steps = list(steps)
        self._index = 0

    def init_chat(self):
        return _FakeMsg(self._init_content)

    def step(self, _input_msg):
        assistant_response, user_response = self._steps[self._index]
        self._index += 1
        return assistant_response, user_response


def _resp(content=None, terminated=False):
    msg = _FakeMsg(content) if content is not None else None
    return _FakeChatAgentResponse(msg=msg, terminated=terminated)


class ChainTopologyRunnerTests(unittest.TestCase):
    def test_two_agent_chain_relays_target_text_and_logs_each_speaker(self) -> None:
        runner = CascadeTopologyRunner()
        source = AgentNode("source-id", "Source", hop=0, agent=FakeAgent("source-id"))
        target = AgentNode("target-id", "Target", hop=1, agent=FakeAgent("target-id"))

        session = _FakeSession(
            init_content="INIT",
            steps=[
                (_resp("src-1"), _resp("tgt-1")),
                (_resp("src-2"), _resp("tgt-2 CAMEL_TASK_DONE")),
            ],
        )

        with patch.object(runner_module, "create_role_playing_session", return_value=session):
            result = runner.run_two_agent_chain(
                source=source,
                target=target,
                task_prompt="solve task",
                condition="attack",
                chat_turn_limit=3,
            )

        self.assertEqual(result.topology, "chain_two")
        self.assertEqual(result.metadata["final_relay"], "tgt-2 CAMEL_TASK_DONE")
        self.assertEqual(
            [(m["agent_id"], m["speaker"], m["turn"], m["content"]) for m in result.messages],
            [
                ("source-id", "source_init", 0, "INIT"),
                ("target-id", "target", 0, "tgt-1"),
                ("source-id", "source", 1, "src-1"),
                ("target-id", "target", 1, "tgt-2 CAMEL_TASK_DONE"),
            ],
        )
        # Every logged message produces one uncertainty record.
        self.assertEqual(len(result.uncertainty), len(result.messages))
        self.assertEqual(
            [r["agent_id"] for r in result.uncertainty],
            ["source-id", "target-id", "source-id", "target-id"],
        )

    def test_two_agent_chain_stops_when_target_terminates(self) -> None:
        runner = CascadeTopologyRunner()
        source = AgentNode("source-id", "Source", hop=0, agent=FakeAgent("source-id"))
        target = AgentNode("target-id", "Target", hop=1, agent=FakeAgent("target-id"))

        session = _FakeSession(
            init_content="INIT",
            steps=[(_resp("src-1"), _resp("tgt-1", terminated=True))],
        )

        with patch.object(runner_module, "create_role_playing_session", return_value=session):
            result = runner.run_two_agent_chain(
                source=source,
                target=target,
                task_prompt="solve task",
                condition="baseline",
                chat_turn_limit=5,
            )

        self.assertEqual(
            [(m["speaker"], m["content"]) for m in result.messages],
            [("source_init", "INIT"), ("target", "tgt-1")],
        )
        self.assertEqual(result.metadata["final_relay"], "tgt-1")

    def test_three_agent_chain_threads_midpoint_relay_into_second_pair(self) -> None:
        runner = CascadeTopologyRunner()
        source = AgentNode("source-id", "Source", hop=0, agent=FakeAgent("source-id"))
        middle = AgentNode("middle-id", "Middle", hop=1, agent=FakeAgent("middle-id"))
        target = AgentNode("target-id", "Target", hop=2, agent=FakeAgent("target-id"))

        pair_one = _FakeSession(
            init_content="INIT-1",
            steps=[(_resp("mid-src"), _resp("mid-tgt CAMEL_TASK_DONE"))],
        )
        pair_two = _FakeSession(
            init_content="INIT-2",
            steps=[(_resp("tgt-src"), _resp("final CAMEL_TASK_DONE"))],
        )
        captured_prompts = []

        def fake_session(*_args, **kwargs):
            captured_prompts.append(kwargs["task_prompt"])
            return pair_one if len(captured_prompts) == 1 else pair_two

        with patch.object(runner_module, "create_role_playing_session", side_effect=fake_session):
            result = runner.run_three_agent_chain(
                source=source,
                middle=middle,
                target=target,
                task_prompt="solve task",
                condition="attack",
                chat_turn_limit=2,
            )

        self.assertEqual(result.topology, "chain_three")
        self.assertEqual(result.metadata["midpoint_relay"], "mid-tgt CAMEL_TASK_DONE")
        self.assertEqual(result.metadata["final_relay"], "final CAMEL_TASK_DONE")
        # The second pair's prompt must carry the midpoint relay forward.
        self.assertIn("mid-tgt CAMEL_TASK_DONE", captured_prompts[1])
        self.assertEqual(
            [(m["agent_id"], m["speaker"], m["turn"]) for m in result.messages],
            [
                ("source-id", "source_init", 0),
                ("middle-id", "target", 0),
                ("middle-id", "source_init", 1),
                ("target-id", "target", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
