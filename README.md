# Probabilistic Serial Mechanism Exploration

This repository contains code for simulating a multi-player fair division game under the **Probabilistic Serial (PS)** mechanism. The project checks whether players can benefit from strategic misreporting of preferences by seeking **Pure Nash Equilibria** (PNE) with higher social welfare than the truthful profile.

> **Special Thanks:**  
> We extend our gratitude to **Nick Mattei** for his invaluable guidance and insights on the topics of computational social choice and fair division.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [File Structure](#file-structure)  
6. [Example Results](#example-results)  
7. [License](#license)

---

## Overview

In this project, we:
- **Generate Random Preferences** for each player as permutations of items.
- **Assign Valuations** based on these preferences (higher-preferred items get larger random valuations).
- **Run the Probabilistic Serial Mechanism** to allocate items fractionally among players.
- **Calculate Social Welfare** for the truthful scenario.
- **Enumerate All Strategies** (all permutations of preferences each player might declare).
- **Find Pure Nash Equilibria**, testing if any PNE provides higher social welfare than telling the truth.

The code demonstrates how lying about preferences can (sometimes) yield a strictly better outcome in terms of total utility—but only if it forms a stable strategy profile (i.e., a PNE).

---

## Features

- **Probabilistic Serial Allocation**:  
  Players “simultaneously eat” items based on preference until supplies run out, yielding a fractional allocation.

- **Payoff Matrix Generation**:  
  Explores *every possible* strategy combination for all players.

- **Pure Nash Equilibrium Detection**:  
  Identifies strategy profiles where no single player can unilaterally deviate to improve their utility.

- **Comparative Analysis**:  
  Compares social welfare (sum of players’ utilities) under truthful reporting vs. potential equilibria.

---

## Installation

1. **Clone this repository** (or download the ZIP):
    ```bash
    git clone https://github.com/<username>/<repo_name>.git
    cd <repo_name>
    ```
2. **Install Dependencies**:
    - [Python 3.7+](https://www.python.org/downloads/)  
    - [NumPy](https://pypi.org/project/numpy/)  
    - (Optional) [fractions](https://docs.python.org/3/library/fractions.html) is part of the Python standard library

    You can install NumPy via:
    ```bash
    pip install numpy
    ```

---

## Usage

1. **(Optional) Adjust Parameters**  
   Edit the bottom lines of `algorithm.py` (or whatever you named the file) to set:
   - `NUM_ATTEMPTS` (number of random preference profiles to test)  
   - `NUM_PLAYERS`  
   - `NUM_ITEMS`

2. **Run the Script**:
    ```bash
    python algorithm.py
    ```
   or
    ```bash
    python3 algorithm.py
    ```

3. **Observe the Output**:  
   The script will:
   - Print each attempt’s random valuations.
   - Show the allocation and social welfare of the truthful strategy.
   - Check for any Pure Nash Equilibria with higher social welfare.
   - Stop early if such a PNE is found (or finish all attempts otherwise).

---

## File Structure

