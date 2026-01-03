import itertools
import random
from fractions import Fraction

import numpy as np


# --------------------------------------------------------------------------
# 1. Generate random preference and valuation profiles
# --------------------------------------------------------------------------

def generate_random_preference_profiles(num_players: int, num_items: int):
    """Each player's preference is a random permutation of item indices."""
    return [random.sample(range(num_items), num_items) for _ in range(num_players)]


def generate_random_valuation_profiles(preference_profiles, rng=None):
    """
    Generate valuations consistent with each player's truthful preference ordering:
    draw random values, sort descending, assign along the preference ranking.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_players = len(preference_profiles)
    num_items = len(preference_profiles[0])
    valuations = np.zeros((num_players, num_items), dtype=float)

    for i, prefs in enumerate(preference_profiles):
        values = rng.random(num_items)
        values.sort()
        values = values[::-1]
        for rank, item in enumerate(prefs):
            valuations[i, item] = values[rank]

    return valuations


# --------------------------------------------------------------------------
# 2. Probabilistic Serial Allocation (Simultaneous Eating)
# --------------------------------------------------------------------------

def probabilistic_serial(profile, return_fractions: bool = False):
    """
    Probabilistic Serial (PS): all agents eat their current top remaining item at unit speed.
    Items have unit supply. When an item is exhausted, agents move to their next available item.

    Returns: allocation matrix A where A[i, o] is the fraction of item o allocated to agent i.
    """
    n = len(profile)
    m = len(profile[0])
    # Basic validation (cheap sanity)
    for i in range(n):
        if len(profile[i]) != m:
            raise ValueError("All players must rank the same number of items.")
        if sorted(profile[i]) != list(range(m)):
            raise ValueError("Each player's preference must be a permutation of 0..m-1.")

    supply = [Fraction(1, 1) for _ in range(m)]
    alloc = [[Fraction(0, 1) for _ in range(m)] for _ in range(n)]
    rank = [0] * n

    while True:
        eating = [-1] * n
        count = [0] * m
        active = False

        # Who eats what?
        for i in range(n):
            while rank[i] < m and supply[profile[i][rank[i]]] == 0:
                rank[i] += 1
            if rank[i] < m:
                o = profile[i][rank[i]]
                eating[i] = o
                count[o] += 1
                active = True

        if not active:
            break  # no one can eat anything

        # Time until some currently-eaten item(s) is exhausted
        min_time = min(supply[o] / count[o] for o in range(m) if count[o] > 0)

        # Everyone eats for min_time
        for i, o in enumerate(eating):
            if o != -1:
                alloc[i][o] += min_time

        # Update supplies (consume count[o] * min_time of each eaten item)
        for o in range(m):
            if count[o] > 0:
                supply[o] -= min_time * count[o]
                # supply[o] should hit exactly 0 for at least one o by construction

    if return_fractions:
        return alloc

    return np.array([[float(x) for x in row] for row in alloc], dtype=float)


# --------------------------------------------------------------------------
# 3. Social Welfare and Utilities
# --------------------------------------------------------------------------

def calculate_utilities(valuation_profiles: np.ndarray, allocation: np.ndarray):
    """Utility_i = sum_o v[i,o] * x[i,o]."""
    return np.einsum("ij,ij->i", valuation_profiles, allocation)


def calculate_social_welfare(valuation_profiles: np.ndarray, allocation: np.ndarray):
    """SW = sum_i utility_i."""
    return float(calculate_utilities(valuation_profiles, allocation).sum())


# --------------------------------------------------------------------------
# 4. Payoff table (brute force)
# --------------------------------------------------------------------------

def generate_permutations(num_items: int):
    return list(itertools.permutations(range(num_items)))


def build_payoff_dict(permutations_list, num_players: int, valuation_profiles: np.ndarray):
    """
    payoff[strategy_profile] = utilities vector
    where strategy_profile is a tuple of permutation indices (one per player).
    """
    k = len(permutations_list)
    payoff = {}

    for strat_idx_profile in itertools.product(range(k), repeat=num_players):
        reported_profile = [permutations_list[idx] for idx in strat_idx_profile]
        allocation = probabilistic_serial(reported_profile)
        payoff[strat_idx_profile] = calculate_utilities(valuation_profiles, allocation)

    return payoff


# --------------------------------------------------------------------------
# 5. Pure Nash equilibria
# --------------------------------------------------------------------------

def find_pure_nash_equilibria(payoff: dict, num_players: int, num_strategies: int, eps: float = 1e-12):
    equilibria = []
    for profile, utils in payoff.items():
        is_eq = True
        for p in range(num_players):
            u = utils[p]
            for s in range(num_strategies):
                if s == profile[p]:
                    continue
                dev = list(profile)
                dev[p] = s
                dev = tuple(dev)
                if payoff[dev][p] > u + eps:
                    is_eq = False
                    break
            if not is_eq:
                break
        if is_eq:
            equilibria.append((profile, utils))
    return equilibria


# --------------------------------------------------------------------------
# 6. Main driver: search for a PNE with higher SW than truthful
# --------------------------------------------------------------------------

def find_better_pne_profiles(num_attempts: int, num_players: int, num_items: int,
                            threshold: float = 1e-5, seed: int | None = None):
    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    permutations_list = generate_permutations(num_items)
    k = len(permutations_list)

    for attempt in range(1, num_attempts + 1):
        print(f"\nAttempt {attempt}:\n{'-' * 50}")

        truthful_prefs = generate_random_preference_profiles(num_players, num_items)
        valuations = generate_random_valuation_profiles(truthful_prefs, rng=rng)

        alloc_truth = probabilistic_serial(truthful_prefs)
        sw_truth = calculate_social_welfare(valuations, alloc_truth)

        print("Valuation Profiles:\n", valuations)
        print("Truthful Preference Profiles:")
        for i, prefs in enumerate(truthful_prefs):
            print(f"  Player {i+1}: {prefs}")
        print("\nTruthful Allocation:\n", alloc_truth)
        print("Social Welfare (Truthful):", sw_truth)

        payoff = build_payoff_dict(permutations_list, num_players, valuations)
        equilibria = find_pure_nash_equilibria(payoff, num_players, k)

        for profile_idx, _utils in equilibria:
            reported_profile = [permutations_list[idx] for idx in profile_idx]
            alloc_pne = probabilistic_serial(reported_profile)
            sw_pne = calculate_social_welfare(valuations, alloc_pne)

            if sw_pne - sw_truth > threshold:
                print("\n>>> Found a strictly better PNE. <<<")
                print("Better PNE Social Welfare:", sw_pne)
                print("Better PNE Strategies:")
                for i, strat in enumerate(reported_profile):
                    print(f"  Player {i+1}: {strat}")
                print("PNE Allocation:\n", alloc_pne)
                print("PNE Utilities:", calculate_utilities(valuations, alloc_pne))
                return {
                    "attempt": attempt,
                    "truthful_prefs": truthful_prefs,
                    "valuations": valuations,
                    "sw_truthful": sw_truth,
                    "pne_profile_idx": profile_idx,
                    "pne_reported_profile": reported_profile,
                    "sw_pne": sw_pne,
                }

        print("No better Pure Nash Equilibrium found in this attempt.")

    print("\nNo strictly better Pure Nash Equilibria were found after all attempts.")
    return None


# --------------------------------------------------------------------------
# 7. Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_ATTEMPTS = 100
    NUM_PLAYERS = 3
    NUM_ITEMS = 3

    result = find_better_pne_profiles(NUM_ATTEMPTS, NUM_PLAYERS, NUM_ITEMS, seed=None)
    # result is either None or a dict with the witness profile
