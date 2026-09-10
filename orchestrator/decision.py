from itertools import product

from models.workload import Criticality, ExecutionMode

from orchestrator.feasibility import get_feasible_actions
from orchestrator.utility import calculate_score, calculate_utility
from orchestrator.resource_manager import can_allocate, allocate
from orchestrator.runtime import update_workload_state
from orchestrator.history import classify_adaptation


def _evaluate_action(workload, location, mode, system, allocations):
    """Return the proposed-policy score if the action can be jointly allocated."""
    if not can_allocate(
        workload,
        location,
        mode,
        system,
        allocations
    ):
        return None

    result = calculate_score(
        workload,
        location,
        mode,
        system
    )

    return result


def _critical_workload(workload):
    """Mission-critical workloads are explicitly marked VERY_HIGH."""
    return workload.criticality == Criticality.VERY_HIGH


def _joint_allocate(critical_workloads, action_combination, system):
    """Check whether a complete critical-workload action combination fits."""
    allocations = {}

    for workload, (location, mode) in zip(
        critical_workloads,
        action_combination
    ):
        if not can_allocate(
            workload,
            location,
            mode,
            system,
            allocations
        ):
            return None

        allocate(
            workload,
            location,
            mode,
            allocations
        )

    return allocations


def _select_critical_actions(critical_workloads, system):
    """
    Jointly select actions for mission-critical workloads.

    The objective is lexicographic:
      1. Preserve critical service (never intentionally suspend a critical workload).
      2. Maximize aggregate mission utility of critical workloads.
      3. Maximize the minimum critical-workload quality.
      4. Minimize the aggregate orchestration cost.

    The critical action space is intentionally small in this prototype, so
    exhaustive enumeration is used instead of a heavyweight optimizer.
    """
    action_lists = []

    for workload in critical_workloads:
        feasible = [
            action
            for action in get_feasible_actions(workload, system)
            if action[1] != ExecutionMode.SUSPENDED
        ]

        if not feasible:
            action_lists.append([])
        else:
            action_lists.append(feasible)

    if any(not actions for actions in action_lists):
        return None, {}

    best = None

    for combination in product(*action_lists):
        allocations = _joint_allocate(
            critical_workloads,
            combination,
            system
        )

        if allocations is None:
            continue

        total_utility = 0.0
        quality_values = []
        total_cost = 0.0
        results = []

        for workload, (location, mode) in zip(
            critical_workloads,
            combination
        ):
            result = calculate_score(
                workload,
                location,
                mode,
                system
            )

            utility = calculate_utility(
                workload,
                mode,
                system.mission.phase
            )

            total_utility += utility
            quality_values.append(utility / max(
                workload.criticality.value * 20,
                1
            ))
            total_cost += result["score"] - utility
            results.append(result)

        min_quality = min(quality_values)

        # Higher utility first; then protect the weakest critical workload;
        # finally prefer lower cost. Tuple is ordered for max().
        candidate_key = (
            total_utility,
            min_quality,
            -total_cost
        )

        if best is None or candidate_key > best["key"]:
            best = {
                "key": candidate_key,
                "combination": combination,
                "allocations": allocations,
                "results": results
            }

    if best is None:
        return None, {}

    return best, best["allocations"]


def _commit_decision(workload, location, mode, result, allocations, decisions):
    """Apply a selected action and append a standard decision record."""
    old_location = workload.current_location
    old_mode = workload.current_mode

    update_workload_state(
        workload,
        location,
        mode
    )

    allocate(
        workload,
        location,
        mode,
        allocations
    )

    adapted = (
        old_location != location
        or old_mode != mode
    )

    adaptation_type = classify_adaptation(
        old_location.value,
        old_mode.value,
        location.value,
        mode.value
    )

    decisions.append({
        "workload": workload.name,
        "location": location,
        "mode": mode,
        "score": result["score"],
        "details": result,
        "adaptation_type": adaptation_type,
        "adapted": adapted
    })


def _append_infeasible(workload, decisions):
    decisions.append({
        "workload": workload.name,
        "location": None,
        "mode": None,
        "score": None,
        "adapted": False
    })


def select_actions_globally(system):
    """
    Mission-aware adaptive orchestration policy.

    Unlike the old purely sequential greedy policy, mission-critical
    workloads are selected jointly first. This prevents lower-priority
    workloads from consuming resources that are needed to preserve critical
    functionality.

    After critical workloads are protected, remaining workloads are selected
    using the existing utility/cost scoring mechanism.
    """
    allocations = {}
    decisions = []

    critical_workloads = [
        workload
        for workload in system.workloads
        if _critical_workload(workload)
    ]

    noncritical_workloads = [
        workload
        for workload in system.workloads
        if not _critical_workload(workload)
    ]

    # ------------------------------------------------------------
    # PHASE 1: Joint protection of mission-critical workloads
    # ------------------------------------------------------------
    critical_selection, critical_allocations = _select_critical_actions(
        critical_workloads,
        system
    )

    if critical_selection is not None:
        # Recreate the critical allocation map by committing each action.
        for workload, (location, mode), result in zip(
            critical_workloads,
            critical_selection["combination"],
            critical_selection["results"]
        ):
            _commit_decision(
                workload,
                location,
                mode,
                result,
                allocations,
                decisions
            )
    else:
        # If no joint feasible critical combination exists, fall back to the
        # old criticality-ordered greedy mechanism. This represents a true
        # physical/resource infeasibility rather than sacrificing a critical
        # workload because a lower-priority workload was allocated first.
        for workload in sorted(
            critical_workloads,
            key=lambda w: w.criticality.value,
            reverse=True
        ):
            feasible_actions = get_feasible_actions(
                workload,
                system
            )

            candidates = []

            for location, mode in feasible_actions:
                if mode == ExecutionMode.SUSPENDED:
                    continue

                result = _evaluate_action(
                    workload,
                    location,
                    mode,
                    system,
                    allocations
                )

                if result is not None:
                    candidates.append((location, mode, result))

            if not candidates:
                _append_infeasible(workload, decisions)
                continue

            best = max(
                candidates,
                key=lambda x: x[2]["score"]
            )

            _commit_decision(
                workload,
                best[0],
                best[1],
                best[2],
                allocations,
                decisions
            )

    # ------------------------------------------------------------
    # PHASE 2: Allocate remaining resources to non-critical work
    # ------------------------------------------------------------
    for workload in sorted(
        noncritical_workloads,
        key=lambda w: w.criticality.value,
        reverse=True
    ):
        feasible_actions = get_feasible_actions(
            workload,
            system
        )

        candidates = []

        for location, mode in feasible_actions:
            result = _evaluate_action(
                workload,
                location,
                mode,
                system,
                allocations
            )

            if result is None:
                continue

            candidates.append((location, mode, result))

        if not candidates:
            _append_infeasible(workload, decisions)
            continue

        # Phase 2 is still mission-aware: first maximize mission utility
        # (service quality weighted by the current mission phase). Only when
        # two actions provide the same utility do we use the orchestration
        # score as the efficiency tie-breaker.
        #
        # This prevents the policy from selecting an unnecessarily degraded
        # service merely because it has a lower resource/latency cost.
        best = max(
            candidates,
            key=lambda x: (
                calculate_utility(
                    workload,
                    x[1],
                    system.mission.phase
                ),
                x[2]["score"]
            )
        )

        _commit_decision(
            workload,
            best[0],
            best[1],
            best[2],
            allocations,
            decisions
        )

    return decisions
