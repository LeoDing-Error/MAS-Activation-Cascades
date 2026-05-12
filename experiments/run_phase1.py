from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.cascade_analyzer import CascadeAnalyzer, generate_report
from src.backends.camel_integration import (
    AgentSpec,
    create_chat_agent,
    create_clean_chat_agent,
    create_openai_compatible_agent,
)
from src.experiments.phase1_config import (
    FALLBACK_MODEL,
    HUMANEVAL_SUBSET,
    PRIMARY_MODEL,
    EvalTask,
    default_tasks_for_experiment,
    parse_csv_list,
    parse_int_csv,
    select_tasks,
)
from src.topologies.runner import AgentNode, CascadeTopologyRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase 1 cascade experiments")
    parser.add_argument("--experiment", required=True, choices=["1.1", "1.2", "1.3", "1.4"])
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL)
    parser.add_argument("--steering-vector", type=Path, required=True)
    parser.add_argument("--steering-strength", type=float, default=1.0)
    parser.add_argument("--alphas", default="0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--task-names", default=None)
    parser.add_argument("--task-indices", default=None)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-turn-limit", type=int, default=2)
    parser.add_argument("--clean-api-base", default=None)
    parser.add_argument("--clean-api-key", default="EMPTY")
    parser.add_argument(
        "--allow-local-clean-models",
        action="store_true",
        help="Allow multi-agent runs to instantiate clean HuggingFace models locally instead of requiring an OpenAI-compatible server",
    )
    return parser


def build_agent(
    *,
    agent_id: str,
    role_name: str,
    hop: int,
    model_name: str,
    max_new_tokens: int,
    steering_vector: Path | None = None,
    steering_strength: float = 1.0,
    steering_enabled: bool = False,
    clean_api_base: str | None = None,
    clean_api_key: str = "EMPTY",
) -> AgentNode:
    spec = AgentSpec(
        agent_id=agent_id,
        role_name=role_name,
        system_message=f"You are the {role_name}. Keep responses concise and focused on the current software task.",
        backend_kwargs={"max_new_tokens": max_new_tokens},
    )
    if steering_enabled:
        agent = create_chat_agent(
            spec,
            model_name=model_name,
            steering_vector_path=str(steering_vector),
            steering_strength=steering_strength,
            steering_enabled=True,
        )
    elif clean_api_base:
        agent = create_openai_compatible_agent(
            spec,
            model_name=model_name,
            api_base_url=clean_api_base,
            api_key=clean_api_key,
            model_config_dict={"max_tokens": max_new_tokens},
        )
    else:
        agent = create_clean_chat_agent(spec, model_name=model_name)
    return AgentNode(agent_id=agent_id, role_name=role_name, hop=hop, agent=agent)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_tasks(args: argparse.Namespace) -> List[EvalTask]:
    task_names = parse_csv_list(args.task_names)
    task_indices = parse_int_csv(args.task_indices)
    if args.experiment == "1.1":
        return select_tasks(
            HUMANEVAL_SUBSET,
            task_names=task_names,
            task_indices=task_indices,
            n_tasks=args.n_tasks,
        )
    if task_names is None and task_indices is None and args.n_tasks is None:
        return default_tasks_for_experiment(args.experiment)
    return select_tasks(
        HUMANEVAL_SUBSET,
        task_names=task_names,
        task_indices=task_indices,
        n_tasks=args.n_tasks,
    )


def _save_pairwise_outputs(
    *,
    output_dir: Path,
    prefix: str,
    baseline: Dict[str, Any],
    attack: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    save_json(output_dir / f"{prefix}_baseline.json", baseline)
    save_json(output_dir / f"{prefix}_attack.json", attack)
    save_json(output_dir / f"{prefix}_summary.json", summary)
    generate_report(attack, baseline, output_path=output_dir / "report.txt")


def run_experiment_1_1(args: argparse.Namespace) -> None:
    tasks = resolve_tasks(args)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    all_results: Dict[str, Any] = {"experiment": "1.1", "runs": []}
    baseline_payload: Dict[str, Any] | None = None
    analyzer = CascadeAnalyzer(metric_name="mean_token_entropy", epsilon=0.05)
    summary: Dict[str, Any] = {"experiment": "1.1", "alpha_comparisons": {}}

    for alpha in alphas:
        combined_messages: List[Dict[str, Any]] = []
        combined_uncertainty: List[Dict[str, Any]] = []
        for task in tasks:
            runner = CascadeTopologyRunner()
            agent = build_agent(
                agent_id=f"single_{task.name}",
                role_name="steered_implementer",
                hop=0,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                steering_vector=args.steering_vector,
                steering_strength=alpha,
                steering_enabled=alpha > 0.0,
            )
            result = runner.run_single_agent(
                agent=agent,
                task_prompt=task.prompt,
                condition=f"alpha_{alpha}",
            )
            payload = result.to_dict()
            payload["task"] = asdict(task)
            all_results["runs"].append(payload)
            combined_messages.extend(payload["messages"])
            combined_uncertainty.extend(payload["uncertainty"])

        aggregate = {
            "topology": "single",
            "condition": f"alpha_{alpha}",
            "task_prompt": "multiple_tasks",
            "messages": combined_messages,
            "uncertainty": combined_uncertainty,
            "metadata": {"alpha": alpha},
        }
        if alpha == 0.0:
            baseline_payload = aggregate
        elif baseline_payload is not None:
            summary["alpha_comparisons"][str(alpha)] = asdict(analyzer.summarize(aggregate, baseline_payload))

    output_dir = args.results_dir / "exp1_1"
    save_json(output_dir / "exp1_1_results.json", all_results)
    save_json(output_dir / "exp1_1_summary.json", summary)


def run_experiment_1_2(args: argparse.Namespace) -> None:
    tasks = resolve_tasks(args)
    output_dir = args.results_dir / "exp1_2"
    summaries: List[Dict[str, Any]] = []

    for task in tasks:
        baseline_runner = CascadeTopologyRunner()
        baseline_source = build_agent(
            agent_id="a0",
            role_name="implementer",
            hop=0,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            clean_api_base=args.clean_api_base,
            clean_api_key=args.clean_api_key,
        )
        baseline_target = build_agent(
            agent_id="a1",
            role_name="reviewer",
            hop=1,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            clean_api_base=args.clean_api_base,
            clean_api_key=args.clean_api_key,
        )
        baseline = baseline_runner.run_two_agent_chain(
            source=baseline_source,
            target=baseline_target,
            task_prompt=task.prompt,
            condition="baseline",
            chat_turn_limit=args.chat_turn_limit,
        )

        attack_runner = CascadeTopologyRunner()
        attack_source = build_agent(
            agent_id="a0",
            role_name="implementer",
            hop=0,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            steering_vector=args.steering_vector,
            steering_strength=args.steering_strength,
            steering_enabled=True,
        )
        attack_target = build_agent(
            agent_id="a1",
            role_name="reviewer",
            hop=1,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            clean_api_base=args.clean_api_base,
            clean_api_key=args.clean_api_key,
        )
        attack = attack_runner.run_two_agent_chain(
            source=attack_source,
            target=attack_target,
            task_prompt=task.prompt,
            condition="attack",
            chat_turn_limit=args.chat_turn_limit,
        )

        analyzer = CascadeAnalyzer(metric_name="mean_token_entropy", epsilon=0.05)
        summary = asdict(analyzer.summarize(attack.to_dict(), baseline.to_dict()))
        task_dir = output_dir / task.name
        _save_pairwise_outputs(
            output_dir=task_dir,
            prefix="exp1_2",
            baseline=baseline.to_dict(),
            attack=attack.to_dict(),
            summary=summary,
        )
        summaries.append({"task": asdict(task), "summary": summary})

        if len(tasks) == 1:
            _save_pairwise_outputs(
                output_dir=output_dir,
                prefix="exp1_2",
                baseline=baseline.to_dict(),
                attack=attack.to_dict(),
                summary=summary,
            )

    save_json(output_dir / "exp1_2_runs.json", {"experiment": "1.2", "runs": summaries})
    save_json(output_dir / "exp1_2_summary.json", {"experiment": "1.2", "task_summaries": summaries})


def run_experiment_1_3(args: argparse.Namespace) -> None:
    tasks = resolve_tasks(args)
    output_dir = args.results_dir / "exp1_3"
    summaries: List[Dict[str, Any]] = []

    for task in tasks:
        baseline_runner = CascadeTopologyRunner()
        baseline = baseline_runner.run_three_agent_chain(
            source=build_agent(
                agent_id="a0",
                role_name="planner",
                hop=0,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                clean_api_base=args.clean_api_base,
                clean_api_key=args.clean_api_key,
            ),
            middle=build_agent(
                agent_id="a1",
                role_name="implementer",
                hop=1,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                clean_api_base=args.clean_api_base,
                clean_api_key=args.clean_api_key,
            ),
            target=build_agent(
                agent_id="a2",
                role_name="reviewer",
                hop=2,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                clean_api_base=args.clean_api_base,
                clean_api_key=args.clean_api_key,
            ),
            task_prompt=task.prompt,
            condition="baseline",
            chat_turn_limit=args.chat_turn_limit,
        )

        attack_runner = CascadeTopologyRunner()
        attack = attack_runner.run_three_agent_chain(
            source=build_agent(
                agent_id="a0",
                role_name="planner",
                hop=0,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                steering_vector=args.steering_vector,
                steering_strength=args.steering_strength,
                steering_enabled=True,
            ),
            middle=build_agent(
                agent_id="a1",
                role_name="implementer",
                hop=1,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                clean_api_base=args.clean_api_base,
                clean_api_key=args.clean_api_key,
            ),
            target=build_agent(
                agent_id="a2",
                role_name="reviewer",
                hop=2,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                clean_api_base=args.clean_api_base,
                clean_api_key=args.clean_api_key,
            ),
            task_prompt=task.prompt,
            condition="attack",
            chat_turn_limit=args.chat_turn_limit,
        )

        analyzer = CascadeAnalyzer(metric_name="mean_token_entropy", epsilon=0.05)
        summary = asdict(analyzer.summarize(attack.to_dict(), baseline.to_dict()))
        task_dir = output_dir / task.name
        _save_pairwise_outputs(
            output_dir=task_dir,
            prefix="exp1_3",
            baseline=baseline.to_dict(),
            attack=attack.to_dict(),
            summary=summary,
        )
        summaries.append({"task": asdict(task), "summary": summary})

        if len(tasks) == 1:
            _save_pairwise_outputs(
                output_dir=output_dir,
                prefix="exp1_3",
                baseline=baseline.to_dict(),
                attack=attack.to_dict(),
                summary=summary,
            )

    save_json(output_dir / "exp1_3_runs.json", {"experiment": "1.3", "runs": summaries})
    save_json(output_dir / "exp1_3_summary.json", {"experiment": "1.3", "task_summaries": summaries})


def run_experiment_1_4(args: argparse.Namespace) -> None:
    tasks = resolve_tasks(args)
    output_dir = args.results_dir / "exp1_4"

    peripheral_roles = ["frontend", "backend", "testing"]
    summaries: List[Dict[str, Any]] = []

    for task in tasks:
        baseline_runner = CascadeTopologyRunner()
        baseline = baseline_runner.run_star_topology(
            hub=build_agent(
                agent_id="hub",
                role_name="hub",
                hop=0,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                clean_api_base=args.clean_api_base,
                clean_api_key=args.clean_api_key,
            ),
            peripherals=[
                build_agent(
                    agent_id=f"leaf_{role}",
                    role_name=role,
                    hop=1,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    clean_api_base=args.clean_api_base,
                    clean_api_key=args.clean_api_key,
                )
                for role in peripheral_roles
            ],
            task_prompt=task.prompt,
            condition="baseline",
        )

        attack_runner = CascadeTopologyRunner()
        attack = attack_runner.run_star_topology(
            hub=build_agent(
                agent_id="hub",
                role_name="hub",
                hop=0,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                steering_vector=args.steering_vector,
                steering_strength=args.steering_strength,
                steering_enabled=True,
            ),
            peripherals=[
                build_agent(
                    agent_id=f"leaf_{role}",
                    role_name=role,
                    hop=1,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    clean_api_base=args.clean_api_base,
                    clean_api_key=args.clean_api_key,
                )
                for role in peripheral_roles
            ],
            task_prompt=task.prompt,
            condition="attack",
        )

        analyzer = CascadeAnalyzer(metric_name="mean_token_entropy", epsilon=0.05)
        summary = asdict(analyzer.summarize(attack.to_dict(), baseline.to_dict()))
        task_dir = output_dir / task.name
        _save_pairwise_outputs(
            output_dir=task_dir,
            prefix="exp1_4",
            baseline=baseline.to_dict(),
            attack=attack.to_dict(),
            summary=summary,
        )
        summaries.append({"task": asdict(task), "summary": summary})

        if len(tasks) == 1:
            _save_pairwise_outputs(
                output_dir=output_dir,
                prefix="exp1_4",
                baseline=baseline.to_dict(),
                attack=attack.to_dict(),
                summary=summary,
            )

    save_json(output_dir / "exp1_4_runs.json", {"experiment": "1.4", "runs": summaries})
    save_json(output_dir / "exp1_4_summary.json", {"experiment": "1.4", "task_summaries": summaries})


def main() -> None:
    args = _build_parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in {"1.2", "1.3", "1.4"} and args.clean_api_base is None and not args.allow_local_clean_models:
        raise ValueError(
            "Experiments 1.2-1.4 require --clean-api-base by default to avoid loading multiple clean model copies in one process. "
            "Run scripts/serve_clean_model.sh and pass --clean-api-base, or override with --allow-local-clean-models for small-model smoke tests."
        )

    if args.experiment == "1.1":
        run_experiment_1_1(args)
    elif args.experiment == "1.2":
        run_experiment_1_2(args)
    elif args.experiment == "1.3":
        run_experiment_1_3(args)
    elif args.experiment == "1.4":
        run_experiment_1_4(args)
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")


if __name__ == "__main__":
    main()
