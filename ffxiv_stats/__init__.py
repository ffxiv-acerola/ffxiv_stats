"""
Compute damage variability in the critically acclaimed MMORPG Final Fantasy XIV.

Account for variability from hit types, random damage rolls, and rotations.
This module computes exact damage DPS distributions or the first three moments (mean, variance, skewness).

"""

from ffxiv_stats import moments, rate, jobs, modifiers
from ffxiv_stats.moments import Rotation
from ffxiv_stats.rate import Rate
