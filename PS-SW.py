import sys
from fractions import Fraction
import numpy as np
import random
import itertools

# --------------------------------------------------------------------------
# 1. Generate random preference and valuation profiles
# --------------------------------------------------------------------------

def generate_random_preference_profiles(num_players, num_items):
    """
    Generate random preference profiles for each player.
    Each player's preference profile is a random permutation of item indices.
    """
    return [random.sample(range(num_items), num_items) for _ in range(num_players)]


def generate_random_valuation_profiles(preference_profiles):
    """
    Generate valuation profiles based on the given preference profiles.
    
    - For each player, generate random values in (0, 1), sort them in descending order,
      then assign these valuations according to the player's preference ordering.
    """
    num_players = len(preference_profiles)
    num_items = len(preference_profiles[0])
    valuation_profiles = np.zeros((num_players, num_items))

    for i, prefs in enumerate(preference_profiles):
        values = np.random.rand(num_items)
        values.sort()
        values = values[::-1]
        for rank, item in enumerate(prefs):
            valuation_profiles[i, item] = values[rank]

    return valuation_profiles


# --------------------------------------------------------------------------
# 2. Probabilistic Serial Allocation
# --------------------------------------------------------------------------

def probabilistic_serial(profile):
    """
    Implement the probabilistic serial mechanism for a given preference profile.
    Returns an allocation matrix where allocation[i, o] is the fraction of item o
    allocated to player i.
    """
    num_players = len(profile)
    num_items = len(profile[0])

    supply = {o: Fraction(1, 1) for o in range(num_items)}
    allocation = {(i, o): Fraction(0, 1) for i in range(num_players) for o in range(num_items)}
    current_rank = [0] * num_players

    while any(supply[o] > 0 for o in range(num_items)):
        eating_map = {}
        eaters_count = {o: 0 for o in range(num_items)}

        # Assign each player the item they're currently "eating"
        for i in range(num_players):
            while current_rank[i] < num_items and supply[profile[i][current_rank[i]]] == 0:
                current_rank[i] += 1
            if current_rank[i] < num_items:
                item_eating = profile[i][current_rank[i]]
                eating_map[i] = item_eating
                eaters_count[item_eating] += 1

        if not eating_map:  # No one can eat if items are depleted
            break

        # Find how long until next supply runs out
        min_time = min(
            supply[o] / eaters_count[o]
            for o in range(num_items)
            if eaters_count[o] > 0
        )

        # Update allocations
        for i, item in eating_map.items():
            allocation[(i, item)] += min_time
            supply[item] -= min_time
            if supply[item] == 0:
                current_rank[i] += 1

    # Convert to NumPy array
    allocation_matrix = np.array([
        [float(allocation[(i, o)]) for o in range(num_items)]
        for i in range(num_players)
    ])
    return allocation_matrix


# --------------------------------------------------------------------------
# 3. Social Welfare and Utilities
# --------------------------------------------------------------------------

def calculate_social_welfare(valuation_profiles, allocation):
    """
    Calculate the social welfare as the sum of valuations * allocations.
    """
    return np.sum(np.multiply(valuation_profiles, allocation))


def calculate_utilities(valuation_profiles, allocation):
    """
    Calculate each player's utility as the sum of valuations * allocations per item.
    """
    return np.sum(np.multiply(valuation_profiles, allocation), axis=1)


# --------------------------------------------------------------------------
# 4. Generate permutations & payoff matrix
# --------------------------------------------------------------------------

def generate_permutations(num_items):
    """
    Generate all permutations of num_items as potential strategies.
    """
    return list(itertools.permutations(range(num_items)))


def generate_payoff_matrix(permutations_list, num_players, valuation_profiles):
    """
    Build a payoff matrix for all strategy combinations (Cartesian product).
    Each row is: (strategy_idx_for_player1, ..., strategy_idx_for_playerN,
                  utility_player1, ..., utility_playerN).
    """
    strategy_combinations = list(itertools.product(permutations_list, repeat=num_players))
    payoff_matrix = []

    # For quick lookup of permutation -> index
    permutation_to_index = {perm: idx for idx, perm in enumerate(permutations_list)}

    for strategies in strategy_combinations:
        allocation = probabilistic_serial(strategies)
        utilities = calculate_utilities(valuation_profiles, allocation)
        strategy_indices = [permutation_to_index[strat] for strat in strategies]
        payoff_matrix.append((*strategy_indices, *utilities))

    return payoff_matrix, strategy_combinations


# --------------------------------------------------------------------------
# 5. Finding Pure Nash Equilibria
# --------------------------------------------------------------------------

def find_pure_nash_equilibria(payoff_matrix, num_players, num_strategies):
    """
    Identify all pure Nash equilibria:
    No single player can deviate to another strategy and gain strictly higher utility.
    """
    strategy_profile_dict = {}
    for entry in payoff_matrix:
        strategy_indices = entry[:num_players]
        utilities = entry[num_players:]
        strategy_profile_dict[tuple(strategy_indices)] = utilities

    equilibria = []
    for entry in payoff_matrix:
        strategy_indices = entry[:num_players]
        utilities = entry[num_players:]
        is_equilibrium = True

        for p in range(num_players):
            current_utility = utilities[p]
            for s in range(num_strategies):
                if s != strategy_indices[p]:
                    deviated_indices = list(strategy_indices)
                    deviated_indices[p] = s
                    deviated_indices = tuple(deviated_indices)
                    deviated_utils = strategy_profile_dict.get(deviated_indices)

                    if deviated_utils is not None and deviated_utils[p] > current_utility:
                        is_equilibrium = False
                        break
            if not is_equilibrium:
                break

        if is_equilibrium:
            equilibria.append(entry)
    return equilibria


# --------------------------------------------------------------------------
# 6. Main driver: exit as soon as a better PNE is found
# --------------------------------------------------------------------------

def find_better_pne_profiles(num_attempts, num_players, num_items):
    """
    Generate random preference/valuation profiles multiple times. For each attempt:
      1) Compute the truthful scenario (allocation + social welfare).
      2) Build the payoff matrix & identify all PNE.
      3) If any PNE is strictly better by more than THRESHOLD, exit immediately.
    """
    THRESHOLD = 1e-5

    permutations_list = generate_permutations(num_items)
    num_strategies = len(permutations_list)

    for attempt in range(num_attempts):
        print(f"\nAttempt {attempt + 1}:\n{'-' * 50}")

        # 1) Generate random preferences and valuations
        preference_profiles = generate_random_preference_profiles(num_players, num_items)
        valuation_profiles = generate_random_valuation_profiles(preference_profiles)

        print("Valuation Profiles:\n", valuation_profiles)

        allocation_truthful = probabilistic_serial(preference_profiles)
        sw_truthful = calculate_social_welfare(valuation_profiles, allocation_truthful)

        print("Truthful Preference Profiles:")
        for i, prefs in enumerate(preference_profiles):
            print(f"  Player {i + 1}: {prefs}")

        print("\nTruthful Allocation:\n", allocation_truthful)
        print("Social Welfare (Truthful):", sw_truthful)

        # 2) Build payoff matrix and find PNE
        payoff_matrix, strategy_combinations = generate_payoff_matrix(
            permutations_list, num_players, valuation_profiles
        )
        pure_nash_equilibria = find_pure_nash_equilibria(payoff_matrix, num_players, num_strategies)

        # 3) Check if any PNE is "strictly better" than truthful by more than THRESHOLD
        for eq in pure_nash_equilibria:
            eq_strategies = eq[:num_players]
            strategies = [permutations_list[idx] for idx in eq_strategies]

            allocation_pne = probabilistic_serial(strategies)
            sw_pne = calculate_social_welfare(valuation_profiles, allocation_pne)

            if (sw_pne - sw_truthful) > THRESHOLD:
                # Found a better PNE: print info and exit
                print("\n>>> Found a strictly better PNE. Exiting. <<<")
                print("Better PNE Social Welfare:", sw_pne)
                print("Better PNE Strategies:")
                for i, strat in enumerate(strategies):
                    print(f"  Player {i+1}: {strat}")
                print("PNE Allocation:\n", allocation_pne)
                print("PNE Utilities:", calculate_utilities(valuation_profiles, allocation_pne))

                # Uncomment one of the two lines below:
                # 1) Return from the function (doesn't necessarily kill entire script)
                # return True  

                # 2) Exit the Python interpreter entirely
                sys.exit(0)

        print("No better Pure Nash Equilibrium found in this attempt.")

    # If we finish all attempts without finding a better PNE:
    print("\nNo strictly better Pure Nash Equilibria were found after all attempts.")
    sys.exit(0)  # or return False


# --------------------------------------------------------------------------
# 7. Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: fix seeds for reproducibility
    # random.seed(42)
    # np.random.seed(42)

    NUM_ATTEMPTS = 100
    NUM_PLAYERS = 3
    NUM_ITEMS = 3
    
    find_better_pne_profiles(NUM_ATTEMPTS, NUM_PLAYERS, NUM_ITEMS)
