# ffxiv_stats

## Introduction

`ffxiv_stats` is a Python package to compute statistics relating to damage variability in Final Fantasy XIV. Variability from hit types (critical, direct, critical-direct) and random +/- 5% rolls are considered. Either moments (mean, variance, and skewness) or damage distributions can be calculated. Both methods are exact or (asymptotically exact) and do not rely on sampling.

IMPORTANT: This package is still in the early stages of development and still some sharp edges. It is perfectly usable in its current state, but there is is effectively no error checking/handling. There are also no safety rails; if you try to model a rotation that is impossible in-game, you will still get mathematically correct values. Garbage in = garbage out. Also, be aware that class and method names changes are likely.

IMPORTANT: The effects of hit type rate buffs on skills with guaranteed critical/direct hits is currently not implemented.

ALSO IMPORTANT: Everything here is assuming level 90. There is currently no easy way to handle lower levels.

## Getting started

### Basic usage

Variability can be computed using either the `Rotation` class or one of the role classes (`Healer`, `Tank`, `Melee`, etc.). The `Rotation` class computes variability when `d2` values are known. The role classes inherits the `Rotation` class and converts potencies to `d2` values based on supplied stats. Each role class varies in how it assigns main stats, traits, attack modifier, etc.

### Using the `Rotation` class

The rotation is supplied as a Pandas DataFrame with columns:

* `d2`: Damage of an action before hit type and damage roll variability. 
* `n`: Number of hits for each unique action. Note unique actions depend on `buffs`, `p`, and `l_c`. Action A with a 10% damage buff 10% increase to critical hit rate is different is different than action A with only a 10% buff.
* `p`: list of hit type probabilities in order `[p_NH, p_CH, p_DH, p_CDH]`.
* `l_c`: critical hit damage modifier, should be O(1000).
* `buffs`: List of any buffs present. A 10% damage buff would be `[1.10]`. If no buffs are present, then an empty list `[]`, list with none (`[None]`), or `[1]` can be supplied.
* `is-dot`: boolean for whether the action is a DoT effect. DoT effects have a different support than direct damage.
* `action-name`: name of the action. See Action Naming for more info on how to name actions.

### Using a role class

Using a role class is recommended to go from potencies to d2 values given various stats. Attributes like `main_stat`, `trait`, etc are automatically set to the corresponding values of each role. Rotations are attached using the `attach_rotation`, which inherits the `Rotation` class. However, the `rotation_df` argument is similar to the above dataframe, but does have slightly different columns

* `action-name`: list of actions.
* `potency`: potency of the action
* `p`: list of hit type probabilities in order `[p_NH, p_CH, p_DH, p_CDH]`.
* `l_c`: critical hit damage modifier, should be O(1000).
* `buffs`: List of any buffs present. A 10% damage buff would be `[1.10]`. If no buffs are present, then an empty list `[]`, list with none (`[None]`), or `[1]` can be supplied.
* `damage-type`: str saying the type of damage, `{'direct', 'magic-dot', 'physical-dot', 'auto'}`.
* `main-stat-add`: integer of how much to add to the main stat (used to account for medication).

Instead of a `d2` column, `potency`, `damage_type`, and `main-stat-add` are used together with player stats to compute and add a `d2` column (along with the `is-dot` column). 

### Naming actions

One currently fragile part is how actions are named. In general, action naming convention should follow the form `'{action_name}-{other thing1}_{other_thing2}...'`. This is because when actions are grouped to unique actions and DPS distributions are computed, it is currently done so by taking the unique action name as everything before '-' and ignoring everything after. This will be handled better later.

### Examples

Check out `examples/examples.ipynb` for some basic usages.

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