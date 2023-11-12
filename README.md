# ffxiv_stats

## Introduction

`ffxiv_stats` is a Python package to compute statistics relating to damage variability in Final Fantasy XIV. Variability from hit types (critical, direct, critical-direct) and random +/- 5% rolls are considered. Either moments (mean, variance, and skewness) or damage distributions can be calculated. Both methods are exact or (asymptotically exact) and do not rely on sampling.

IMPORTANT: This package is still in the early stages of development and still some sharp edges. It is perfectly usable in its current state, but there is is effectively no error checking/handling. There are also no safety rails; if you try to model a rotation that is impossible in-game, you will still get mathematically correct values. Garbage in = garbage out. Also, be aware that class and method names changes are likely.

IMPORTANT: The effects of hit type rate buffs on skills with guaranteed critical/direct hits is currently not implemented.

ALSO IMPORTANT: Everything here is assuming level 90. There is currently no easy way to handle lower levels.

## Getting started

### Basic usage

Variability can be computed using either the `Rotation` class or one of the role classes (currently only `Healer` is supported, other roles have not been verified yet). The `Rotation` class computes variability when `d2` values are known. The role classes inherits the `Rotation` class and converts potencies to `d2` values based on supplied stats. Each role class varies in how it assigns main stats, traits, attack modifier, etc.

### Using the `Rotation` class

The rotation is supplied as a Pandas DataFrame with columns and types:

* `action_name`: str, unique name of an action. Unique action depends on `buffs`, `p`, and `l_c` present.
* `base_action`: str, name of an action ignoring buffs. For example, Glare III with chain stratagem and Glare III with mug will have different `action_names`, but the same base_action. Used for grouping actions together.
* `n`: int, number of hits.
* `p`: list of probability lists, in order `[p_NH, p_CH, p_DH, p_CDH]`.
* `d2`: int, base damage value of action before any variability.
* `l_c`: int, damage multiplier for a critical hit. Value should be in the thousands (1250 -> 125% crit buff).
* `buffs`: list of buffs present. A 10% buff should is represented as [1.10]. No buffs can be represented at [1] or None.
* `is_dot`: boolean or 0/1, whether the action is a damage over time effect.

### Using a role class

Using a role class is recommended to go from potencies to d2 values given various stats. Attributes like `main_stat`, `trait`, etc are automatically set to the corresponding values of each role. Rotations are attached using the `attach_rotation`, which inherits the `Rotation` class. However, the `rotation_df` argument is similar to the above dataframe, but does have slightly different columns

* `action_name`: str, unique name of an action. Unique action depends on `buffs`, `p`, and `l_c` present.
* `base_action`: str, name of an action ignoring buffs. For example, Glare III with chain stratagem and Glare III with mug will have different `action_names`, but the same base_action. Used for grouping actions together.
* `potency`: int, potency of the action
* `n`: int, number of hits for the action. 
* `p`: list of probability lists, in order [p_NH, p_CH, p_DH, p_CDH]
* `l_c`: int, damage multiplier for a critical hit. Value should be in the thousands (1250 -> 125% crit buff).
* `buffs`: list of buffs present. A 10% buff should is represented as [1.10]. No buffs can be represented at [1] or None.
* `damage_type`: str saying the type of damage, {'direct', 'magic-dot', 'physical-dot', 'auto'} 
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
* pandas >= 1.2.4
* scipy >= 1.6.3

These are just the versions it was developed with. Specific versions haven't been tested, but `ffxiv_stats` will probably work with lower versions since fairly basic functionalities are used.