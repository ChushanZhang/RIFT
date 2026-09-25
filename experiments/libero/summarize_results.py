"""Aggregate per-task LIBERO JSON files into benchmark summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any


SUITE_ORDER = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def _load_task_results(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result_path in output_dir.glob("*/gpu*_task*_results.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        required = {
            "task_suite",
            "task_id",
            "successes",
            "total_episodes",
            "duration",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{result_path} is missing fields: {sorted(missing)}")
        payload["_path"] = str(result_path)
        results.append(payload)
    return sorted(
        results,
        key=lambda item: (
            SUITE_ORDER.index(item["task_suite"])
            if item["task_suite"] in SUITE_ORDER
            else len(SUITE_ORDER),
            int(item["task_id"]),
        ),
    )


def _suite_summary(task_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_tasks": 0,
            "total_trials": 0,
            "total_successes": 0,
            "total_time": 0.0,
            "max_time": 0.0,
        }
    )
    for result in task_results:
        suite = grouped[str(result["task_suite"])]
        suite["total_tasks"] += 1
        suite["total_trials"] += int(result["total_episodes"])
        suite["total_successes"] += int(result["successes"])
        suite["total_time"] += float(result["duration"])
        suite["max_time"] = max(suite["max_time"], float(result["duration"]))
    for suite in grouped.values():
        suite["success_rate"] = (
            suite["total_successes"] / suite["total_trials"]
            if suite["total_trials"]
            else 0.0
        )
        suite["average_task_time"] = (
            suite["total_time"] / suite["total_tasks"] if suite["total_tasks"] else 0.0
        )
    return dict(grouped)


def summarize_results(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    task_results = _load_task_results(output_dir)
    if not task_results:
        raise FileNotFoundError(
            f"No LIBERO result JSON files found under {output_dir}."
        )

    suites = _suite_summary(task_results)
    total_trials = sum(item["total_trials"] for item in suites.values())
    total_successes = sum(item["total_successes"] for item in suites.values())
    overall = {
        "total_tasks": sum(item["total_tasks"] for item in suites.values()),
        "total_trials": total_trials,
        "total_successes": total_successes,
        "success_rate": total_successes / total_trials if total_trials else 0.0,
        "total_time": sum(item["total_time"] for item in suites.values()),
    }
    summary = {
        "run_id": output_dir.name,
        "suite_stats": suites,
        "task_results": {
            f"{item['task_suite']}_{item['task_id']}": {
                "success_rate": int(item["successes"]) / int(item["total_episodes"]),
                "successes": int(item["successes"]),
                "total_episodes": int(item["total_episodes"]),
                "duration": float(item["duration"]),
                "task_description": item.get("task_description", ""),
            }
            for item in task_results
        },
        "overall": overall,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "task_suite",
                "tasks",
                "trials",
                "successes",
                "success_rate",
                "average_task_time_seconds",
                "max_task_time_seconds",
            ]
        )
        for suite_name in SUITE_ORDER:
            if suite_name not in suites:
                continue
            stats = suites[suite_name]
            writer.writerow(
                [
                    suite_name,
                    stats["total_tasks"],
                    stats["total_trials"],
                    stats["total_successes"],
                    f"{100.0 * stats['success_rate']:.2f}",
                    f"{stats['average_task_time']:.2f}",
                    f"{stats['max_time']:.2f}",
                ]
            )
        writer.writerow(
            [
                "overall",
                overall["total_tasks"],
                overall["total_trials"],
                overall["total_successes"],
                f"{100.0 * overall['success_rate']:.2f}",
                "",
                "",
            ]
        )

    with (output_dir / "task_success_rates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "task_suite",
                "task_id",
                "description",
                "successes",
                "trials",
                "success_rate",
            ]
        )
        for result in task_results:
            trials = int(result["total_episodes"])
            successes = int(result["successes"])
            writer.writerow(
                [
                    result["task_suite"],
                    result["task_id"],
                    result.get("task_description", ""),
                    successes,
                    trials,
                    f"{100.0 * successes / trials:.2f}",
                ]
            )

    print("\nLIBERO benchmark summary")
    for suite_name in SUITE_ORDER:
        if suite_name in suites:
            stats = suites[suite_name]
            print(
                f"{suite_name}: {stats['total_successes']}/{stats['total_trials']} "
                f"({100.0 * stats['success_rate']:.2f}%)"
            )
    print(
        f"overall: {overall['total_successes']}/{overall['total_trials']} "
        f"({100.0 * overall['success_rate']:.2f}%)"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
