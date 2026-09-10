from models.workload import ExecutionMode
from orchestrator.feasibility import get_feasible_actions
from orchestrator.resource_manager import (
    can_allocate,
    allocate,
)
from orchestrator.utility import (
    calculate_latency_cost,
    calculate_energy_cost,
    calculate_resource_cost,
)


ALPHA_LATENCY = 0.40
BETA_ENERGY = 0.30
GAMMA_RESOURCE = 0.30


def calculate_resource_aware_score(
    workload,
    location,
    mode,
    system
):
    """
    Lower score is better.

    The Resource-Aware baseline considers:
    - latency
    - energy
    - resource consumption

    It does NOT consider:
    - mission phase
    - workload criticality
    - mission utility
    """

    latency_cost = calculate_latency_cost(
        workload,
        location,
        mode,
        system
    )

    energy_cost = calculate_energy_cost(
        workload,
        mode
    )

    resource_cost = calculate_resource_cost(
        workload,
        location,
        mode,
        system
    )

    # Normalize the three components.
    latency_normalized = latency_cost / (
        workload.deadline * 20
    )

    energy_normalized = energy_cost / max(
        workload.energy_requirement,
        1
    )

    resource_normalized = (
        workload.compute_requirement
        / max(
            system.onboard.cpu_capacity,
            1
        )
    )

    if mode == ExecutionMode.REDUCED:
        resource_normalized *= 0.5

    elif mode == ExecutionMode.FALLBACK:
        resource_normalized *= 0.2

    elif mode == ExecutionMode.SUSPENDED:
        resource_normalized = 0.0

    score = (
        ALPHA_LATENCY * latency_normalized
        + BETA_ENERGY * energy_normalized
        + GAMMA_RESOURCE * resource_normalized
    )

    return score


def select_resource_aware_action(
    workload,
    system,
    allocations
):
    """
    Select an action for the Resource-Aware baseline.

    Service-level rule:

        Full
          ↓ if no Full action is feasible
        Reduced
          ↓ if no Reduced action is feasible
        Fallback
          ↓ if no non-suspended action is feasible
        Suspended

    Within the highest feasible service level,
    choose the action with the lowest
    resource-aware score.

    This baseline does NOT use:
    - workload criticality
    - mission phase
    - mission utility
    - proposed-policy logic
    """

    feasible_actions = get_feasible_actions(
        workload,
        system
    )

    candidates = []

    for location, mode in feasible_actions:

        if not can_allocate(
            workload,
            location,
            mode,
            system,
            allocations
        ):
            continue

        score = calculate_resource_aware_score(
            workload,
            location,
            mode,
            system
        )

        candidates.append(
            (
                score,
                location,
                mode
            )
        )

    if not candidates:
        return None

    # --------------------------------------------------
    # Prefer the highest feasible service level.
    # --------------------------------------------------

    service_levels = [
        ExecutionMode.FULL,
        ExecutionMode.REDUCED,
        ExecutionMode.FALLBACK,
    ]

    for service_level in service_levels:

        level_candidates = [
            candidate
            for candidate in candidates
            if candidate[2] == service_level
        ]

        if level_candidates:

            # Within the selected service level,
            # minimize latency/energy/resource cost.
            level_candidates.sort(
                key=lambda x: x[0]
            )

            best_score, best_location, best_mode = (
                level_candidates[0]
            )

            allocate(
                workload,
                best_location,
                best_mode,
                allocations
            )

            return best_location, best_mode

    # --------------------------------------------------
    # Suspension is used only when no non-suspended
    # service level is feasible.
    # --------------------------------------------------

    suspended_candidates = [
        candidate
        for candidate in candidates
        if candidate[2] == ExecutionMode.SUSPENDED
    ]

    if suspended_candidates:

        suspended_candidates.sort(
            key=lambda x: x[0]
        )

        best_score, best_location, best_mode = (
            suspended_candidates[0]
        )

        allocate(
            workload,
            best_location,
            best_mode,
            allocations
        )

        return best_location, best_mode

    return None