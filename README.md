# ffxiv_stats

## Introduction

`ffxiv_stats` is a Python package to compute statistics relating to damage variability in Final Fantasy XIV. Variability from hit types (critical, direct, critical-direct) and random +/- 5% rolls are considered. Either moments (mean, variance, and skewness) or damage distributions can be calculated. Both methods are exact and do not rely on sampling.

IMPORTANT: This package is still in the early stages of development and still some sharp edges. It is perfectly usable in its current state, but there is is effectively no error checking/handling. There are also no safety rails; if you try to model a rotation that is impossible in-game, you will still get mathematically correct values. Garbage in = garbage out. Also, be aware that class and method names changes are likely.

## Getting started

### Basic usage

Variability can be computed using either the `Rotation` class or one of the role classes (currently only `Healer` and `Tank` is supported, other roles have not been verified yet). The `Rotation` class computes variability when `d2` values are known. The role classes inherits the `Rotation` class and converts potencies to `d2` values based on supplied stats. Each role class varies in how it assigns main stats, traits, attack modifier, etc.

### Using the `Rotation` class

The rotation is supplied as a Pandas DataFrame with columns and types:

* `action_name`: str, unique name of an action. Unique action depends on `buffs`, `p_i`, and `l_c` present.
* `base_action`: str, name of an action ignoring buffs. For example, Glare III with chain stratagem and Glare III with mug will have different `action_names`, but the same base_action. Used for grouping actions together.
* `n`: int, number of hits.
* `p_n`: probability of a normal hit.
* `p_c`: probability of a critical hit.
* `p_d`: probability of a direct hit.
* `p_cd`: probability of a critical-direct hit.
* `d2`: int, base damage value of action before any variability.
* `l_c`: int, damage multiplier for a critical hit. Value should be in the thousands (1250 -> 125% crit buff).
* `buffs`: Total buff strength, or a list of buffs. A 10% buff should be represented as 1.1. A 5% and 10% buff can be represented as either 1.155 or [1.05, 1.10], but the former is preferred. Saving a dataframe with array columns can be finnicky.
* `is_dot`: boolean or 0/1, whether the action is a damage over time effect.

### Using a role class

Using a role class is recommended to go from potencies to d2 values given various stats. Attributes like `main_stat`, `trait`, etc are automatically set to the corresponding values of each role. Rotations are attached using the `attach_rotation()` method, which inherits the `Rotation` class. However, the `rotation_df` argument is similar to the above dataframe, but does have slightly different columns

* `action_name`: str, unique name of an action. Unique action depends on `buffs`, `p_i`, and `l_c` present.
* `base_action`: str, name of an action ignoring buffs. For example, Glare III with chain stratagem and Glare III with mug will have different `action_names`, but the same base_action. Used for grouping actions together.
* `potency`: int, potency of the action
* `n`: int, number of hits for the action. 
* `p_n`: probability of a normal hit.
* `p_c`: probability of a critical hit.
* `p_d`: probability of a direct hit.
* `p_cd`: probability of a critical-direct hit.
* `l_c`: int, damage multiplier for a critical hit. Value should be in the thousands (1250 -> 125% crit buff).
* `buffs`: Total buff strength, or a list of buffs. A 10% buff should be represented as 1.1. A 5% and 10% buff can be represented as either 1.155 or [1.05, 1.10], but the former is preferred. Saving a dataframe with array columns can be finnicky.
* `damage_type`: str saying the type of damage, {'direct', 'magic-dot', 'physical-dot', 'auto', 'pet'} 
* `main_stat_add`: int, how much to add to the main stat (used to account for medication, if present) when computing d2

Instead of a `d2` column, `potency`, `damage_type`, and `main-stat-add` are used together with player stats to compute and add a `d2` column (along with the `is-dot` column). 

### Examples

Check out `examples/` for some basic usages.

### Installation

`ffxiv_stats` can be installed from source using [flit](https://flit.pypa.io/en/stable/). While in the root directory, use the command

```sh
flit install
```

Alternatively, the package can also be installed with pip.

```
pip install ffxiv_stats
```

### Requirements

The usual scientific computing stack is used:

* numpy >= 1.20.2
* matplotlib >= 3.4.2
* pandas >= 2.0.0
* scipy >= 1.6.3

Aside from `pandas`, specific versions haven't been tested, but `ffxiv_stats` will probably work with lower versions since fairly basic functionalities are used. There are some typing updates present in `pandas` 2.0.0 which do not have backwards compatibility.