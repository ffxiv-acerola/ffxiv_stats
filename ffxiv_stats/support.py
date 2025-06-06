import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from typing import Any


def _coarsened_boundaries(start, end, delta):
    """
    Get the closest start and end points when coarsening a damage support.

    Parameters
    ----------
    start : int
        Uncoarsened starting point
    end : int
        Uncoarsened ending point
    delta : int
        Step size

    Returns
    -------
    coarsened_start : int
        Adjusted starting point that is aligned with delta step size
    coarsened_end : int
        Adjusted ending point that is aligned with delta step size
    """
    rem = start % delta
    if rem > delta // 2:
        coarsened_start = start - rem + delta
    else:
        coarsened_start = start - rem

    rem = end % delta
    if rem > delta // 2:
        coarsened_end = end - rem + delta
    else:
        coarsened_end = end - rem

    return coarsened_start, coarsened_end


class PMFSingle:
    def __init__(
        self,
        d2: int,
        p_n: float,
        p_c: float,
        p_d: float,
        p_cd: float,
        l_c: int,
        is_dot: bool,
        multiplier: float = 1.0,
        p_sum_tol: float = 1e-10,
    ) -> None:
        """One-hit damage distribution for a single possible action.

        Parameters
        ----------
        d2 : int
            Base damage value before any variability
        p_n : float
            Probability of landing a normal hit
        p_c : float
            Probability of landing a critical hit.
        p_d : float
            Probability of landing a direct hit.
        p_cd : float
            Probability of landing a critical-direct hit.
        l_c : int
            Critical hit damage multiplier, expressed as an integer, 1,xxx.
        is_dot : bool
            Whether the attack is a damage over time action.
        multiplier : float, optional
            Damage multiplier due to buffs. Do not include medication., by default 1.0
        p_sum_tol : float, optional
            Tolerance for probability sum check to handle floating point errors, by default 1e-10

        Raises
        ------
        ValueError
            When |p_n + p_c + p_d + p_cd - 1| > p_sum_tol.
        """
        self.l_d = 125
        self.l_c = l_c

        self.d2 = d2
        self.multiplier = multiplier
        self.is_dot = is_dot

        # Normalize probabilities to exactly sum to 1.0
        p_sum = p_n + p_c + p_d + p_cd
        if abs(p_sum - 1.0) > p_sum_tol:
            raise ValueError(
                f"Probabilities are not normalized, p_n + p_c + p_d + p_cd = {p_sum}"
            )

        # Normalize the probabilities to exactly 1.0 to prevent accumulation of errors
        norm_factor = 1.0 / p_sum
        self.p_n = p_n * norm_factor
        self.p_c = p_c * norm_factor
        self.p_d = p_d * norm_factor
        self.p_cd = p_cd * norm_factor

        # Supports are used for computing moments via MGFs
        self.normal_supp = self.get_support("normal")
        self.crit_supp = self.get_support("critical")
        self.dir_supp = self.get_support("direct")
        self.crit_dir_supp = self.get_support("critical-direct")

        # One hit PMF and support used for convolutions
        self.one_hit_pmf, self.one_hit_support = self.get_one_hit_pmf()

    def ch_dmg_modifier(self) -> int:
        """Damage modifier for landing a critical hit.

        Calculates the damage modifier when a critical hit lands by applying
        the critical hit multiplier (l_c) to the base damage (d2).

        Returns
        -------
        int
            The modified damage value after applying critical hit calculations.
        """
        return np.floor(np.floor(self.d2 * self.l_c) / 1000)

    def dh_dmg_modifier(self) -> int:
        """Damage modifier for landing a direct hit.

        Calculates the damage modifier when a direct hit lands by applying
        the direct hit multiplier (l_d) to the base damage (d2).

        Returns
        -------
        int
            The modified damage value after applying direct hit calculations.
        """
        return np.floor(np.floor(self.d2 * self.l_d) / 100)

    def cdh_dmg_modifier(self) -> int:
        """Damage modifier for landing a critical-direct hit.

        Calculates the damage modifier when both a critical hit and direct hit land
        by applying both multipliers (l_c and l_d) to the base damage (d2).

        Returns
        -------
        int
            The modified damage value after applying critical-direct hit calculations.
        """
        ch_damage = np.floor(np.floor(self.d2 * self.l_c) / 1000)
        return np.floor(np.floor(ch_damage * self.l_d) / 100)

    def get_support(self, hit_type) -> np.ndarray:
        """Find the support (all possible damage values) of a single hit.

        Calculates the range of possible damage values for a single ability
        instance based on its base damage (D2), critical hit level (l_c),
        buff multiplier, whether it's a DoT, and the type of hit landed.

        Parameters
        ----------
        hit_type : HitType
            The type of hit landed. Must be one of HitType.NORMAL,
            HitType.CRITICAL, HitType.DIRECT, or HitType.CRITICAL_DIRECT.

        Returns
        -------
        np.ndarray
            An array containing all possible integer damage values for the
            specified hit type after applying buffs and variance.

        Raises
        ------
        ValueError
            If an invalid `hit_type` is provided.
        """
        support = np.array([], dtype=int)  # Initialize support

        if self.is_dot:
            lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2) + 1
            damage_range = np.arange(lower, upper)

            if hit_type == "normal":
                support = np.floor(damage_range * self.multiplier)

            elif hit_type == "critical":
                support = np.floor(
                    np.floor(np.floor(damage_range * self.l_c) / 1000) * self.multiplier
                )

            elif hit_type == "direct":
                support = np.floor(
                    np.floor(np.floor(damage_range * self.l_d) / 100) * self.multiplier
                )

            elif hit_type == "critical-direct":
                ch_dmg = np.floor(np.floor(damage_range * self.l_c) / 1000)
                support = np.floor(
                    np.floor(np.floor(ch_dmg * self.l_d) / 100) * self.multiplier
                )

        # Attack is not a DoT
        else:
            if hit_type == "normal":
                lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2) + 1

            elif hit_type == "critical":
                lower, upper = (
                    np.floor(0.95 * self.ch_dmg_modifier()),
                    np.floor(1.05 * self.ch_dmg_modifier()) + 1,
                )

            elif hit_type == "direct":
                lower, upper = (
                    np.floor(0.95 * self.dh_dmg_modifier()),
                    np.floor(1.05 * self.dh_dmg_modifier()) + 1,
                )

            elif hit_type == "critical-direct":
                lower, upper = (
                    np.floor(0.95 * self.cdh_dmg_modifier()),
                    np.floor(1.05 * self.cdh_dmg_modifier()) + 1,
                )

            damage_range = np.arange(lower, upper)
            support = np.floor(damage_range * self.multiplier)

        # Ensure return type consistency if needed, though floor should yield integers castable floats
        return support.astype(int)

    @staticmethod
    def _get_hit_type_slice(support: np.ndarray, min_roll: int) -> np.ndarray:
        """Calculate the indices for a specific hit type in the combined PMF array.

        Parameters
        ----------
        support : np.ndarray
            Array of possible damage values for a specific hit type.
        min_roll : int
            Minimum damage roll across all hit types.

        Returns
        -------
        np.ndarray
            Array representing the slice indices where this hit type's probabilities should be placed.
        """
        offset = support[0] - min_roll
        hit_range = support - support[0] + offset
        hit_slice = np.arange(hit_range[0], hit_range[-1] + 1, step=1)
        hit_slice = (support - support[0] + offset).astype(int)
        return hit_slice

    def _get_min_roll(self) -> int:
        """Get the lowest possible damage value given **p**.

        Used to keep the 1-hit damage distribution as small as possible for efficient convolutions.
        Mainly used to account for guaranteed hit types or impossible hit types (e.g., no DH melded or wildfire.)


        This assumes that D(nh) < D(dh) < D(ch) < D(cdh), which is always true.

        Returns
        -------
        int
            Lowest possible damage value.
        """
        if self.p_n > 0:
            min_roll = self.normal_supp[0]
        elif self.p_d > 0:
            min_roll = self.dir_supp[0]
        elif self.p_c > 0:
            min_roll = self.crit_supp[0]
        elif self.p_cd:
            min_roll = self.crit_dir_supp[0]
        return min_roll

    def _get_max_roll(self) -> int:
        """Get the largest possible damage value given **p**.

        Used to keep the 1-hit damage distribution as small as possible for efficient convolutions.
        Mainly used to account for guaranteed hit types or impossible hit types (e.g., no DH melded or wildfire.)

        This assumes that D(nh) < D(dh) < D(ch) < D(cdh), which is always true.

        Returns
        -------
        int
            Highest possible damage value.
        """
        if self.p_cd:
            max_roll = self.crit_dir_supp[-1]
        elif self.p_c > 0:
            max_roll = self.crit_supp[-1]
        elif self.p_d > 0:
            max_roll = self.dir_supp[-1]
        elif self.p_n > 0:
            max_roll = self.normal_supp[-1]

        return max_roll

    def get_one_hit_pmf(self) -> tuple[np.ndarray, np.ndarray]:
        """Computes the PMF and support for landing a single hit.

        These quantities are used to compute the N-hit distributions by self-convolution.

        Returns
        -------
        pmf : np.ndarray
            The probability mass function (PMF) for a single hit
        support : np.ndarray
            The support array representing possible damage values
        """
        # Define the bounds of the mixture distribution (lowest roll NH and highest roll CDH)
        # Everything is integers, so the bounds can be defined with an arange
        min_roll = self._get_min_roll()
        max_roll = self._get_max_roll()

        # Mixture distribution defined with multinomial weights
        one_hit_pmf = np.zeros(max_roll - min_roll + 1)
        if self.p_n > 0:
            nh_slice = self._get_hit_type_slice(self.normal_supp, min_roll)
            one_hit_pmf[nh_slice] = self.p_n / self.normal_supp.size
        if self.p_c > 0:
            ch_slice = self._get_hit_type_slice(self.crit_supp, min_roll)
            one_hit_pmf[ch_slice] = self.p_c / self.crit_supp.size
        if self.p_d > 0:
            dh_slice = self._get_hit_type_slice(self.dir_supp, min_roll)
            one_hit_pmf[dh_slice] = self.p_d / self.dir_supp.size
        if self.p_cd > 0:
            cdh_slice = self._get_hit_type_slice(self.crit_dir_supp, min_roll)
            one_hit_pmf[cdh_slice] = self.p_cd / self.crit_dir_supp.size

        one_hit_support = np.arange(min_roll, max_roll + 1, step=1)
        return one_hit_pmf, one_hit_support


class PMFMixture:
    def __init__(
        self,
        d2: list[int],
        w: list[float],
        p_n: float,
        p_c: float,
        p_d: float,
        p_cd: float,
        l_c: int,
        is_dot: bool,
        multiplier: float = 1.0,
        p_sum_tol: float = 1e-10,
    ):
        """Multi-hit damage distribution.

        Parameters
        ----------
        d2 : list[int]
            List of base damage values before any variability
        w : list[float]
            List of weights for each base damage value
        p_n : float
            Probability of landing a normal hit
        p_c : float
            Probability of landing a critical hit
        p_d : float
            Probability of landing a direct hit
        p_cd : float
            Probability of landing a critical-direct hit
        l_c : int
            Critical hit damage multiplier, expressed as an integer, 1,xxx
        is_dot : bool
            Whether the attack is a damage over time action
        multiplier : float, optional
            Damage multiplier due to buffs, by default 1.0
        p_sum_tol : float, optional
            Tolerance for probability sum check to handle floating point errors, by default 1e-10

        Raises
        ------
        ValueError
            When |p_n + p_c + p_d + p_cd - 1| > p_sum_tol
        """
        self.l_d = 125
        self.l_c = l_c

        self.d2 = d2
        self.w = w
        self.multiplier = multiplier
        self.is_dot = is_dot

        # Normalize probabilities to exactly sum to 1.0
        p_sum = p_n + p_c + p_d + p_cd
        if abs(p_sum - 1.0) > p_sum_tol:
            raise ValueError(
                f"Probabilities are not normalized, p_n + p_c + p_d + p_cd = {p_sum}"
            )

        # Normalize the probabilities to exactly 1.0 to prevent accumulation of errors
        norm_factor = 1.0 / p_sum
        self.p_n = p_n * norm_factor
        self.p_c = p_c * norm_factor
        self.p_d = p_d * norm_factor
        self.p_cd = p_cd * norm_factor

        self.one_hit_components: list[PMFSingle] = [
            PMFSingle(
                d,
                p_n,
                p_c,
                p_d,
                p_cd,
                l_c,
                is_dot,
                multiplier,
            )
            for d in d2
        ]

        self.one_hit_pmf, self.one_hit_support = self.mix_single_action_one_hits()
        pass

    def mix_single_action_one_hits(self) -> tuple[np.ndarray, np.ndarray]:
        """Mix multiple one-hit PMFs according to their weights.

        Combines one-hit probability mass functions from multiple damage components
        according to their respective weights to create a mixture distribution.

        Returns
        -------
        mixture_pmf : np.ndarray
            The combined probability mass function
        mixture_support : np.ndarray
            The support array representing possible damage values
        """
        lowest_roll = min([oh.one_hit_support[0] for oh in self.one_hit_components])
        highest_roll = max([oh.one_hit_support[-1] for oh in self.one_hit_components])

        mixture_support = np.arange(lowest_roll, highest_roll + 1, step=1)
        mixture_pmf = np.zeros(shape=mixture_support.shape)

        for idx in range(len(self.d2)):
            mixture_slice = self.one_hit_components[idx].one_hit_support - lowest_roll
            mixture_pmf[mixture_slice] += (
                self.w[idx] * self.one_hit_components[idx].one_hit_pmf
            )

        return mixture_pmf, mixture_support


class ActionDistribution:
    @staticmethod
    def validate_action_dict(
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Validates an action dictionary and extracts validated parameters.

        Parameters
        ----------
        action : dict[str, Any]
            Dictionary containing action parameters

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            A tuple containing:
            1. Validated common parameters (p_n, p_c, etc.)
            2. Validated model-specific parameters (d2, w)

        Raises
        ------
        ValueError
            If required keys are missing or values have invalid types
        TypeError
            If d2 has an unsupported type
        """
        # Validate required keys
        required_keys = ["p_n", "p_c", "p_d", "p_cd", "l_c", "is_dot"]
        for key in required_keys:
            if key not in action:
                raise ValueError(f"Missing required key in action: {key}")

        # Extract and validate common parameters
        validated_params = {
            "p_n": float(action["p_n"]),
            "p_c": float(action["p_c"]),
            "p_d": float(action["p_d"]),
            "p_cd": float(action["p_cd"]),
            "l_c": int(action["l_c"]),
            "is_dot": bool(action["is_dot"]),
            "multiplier": float(action.get("multiplier", action.get("buffs", 1.0))),
            "n_hits": action["n"],
        }

        # Validate and extract d2 and potential weights
        model_params = {}

        if "d2" not in action:
            raise ValueError("Missing required key in action: d2")

        d2_val = action["d2"]

        if isinstance(d2_val, (int, float)):
            # Single d2 value
            model_params["d2"] = int(d2_val)
            model_params["model_type"] = "single"

        elif isinstance(d2_val, (list, tuple)):
            # Multiple d2 values (mixture)
            if "w" not in action:
                raise ValueError("Missing weights 'w' for multiple d2 values")

            w_val = action["w"]
            if not isinstance(w_val, (list, tuple)) or len(w_val) != len(d2_val):
                raise ValueError("Weights 'w' must be a list with same length as 'd2'")

            model_params["d2"] = [int(d) for d in d2_val]
            model_params["w"] = [float(w) for w in w_val]
            model_params["model_type"] = "mixture"

        else:
            raise TypeError(f"d2 must be int or list, got {type(d2_val)}")

        return validated_params, model_params

    def __init__(
        self,
        action: dict[str, Any],
        t: float,
        action_delta: int = 10,
        compute_moments: bool = True,
        compute_distributions: bool = True,
    ):
        """Damage distribution for a single combat action.

        Models the damage distribution of a single combat action which may consist
        of multiple hits.
        """
        self.action = action
        self.t = t
        self.action_delta = action_delta

        # Validate and extract parameters
        validated_params, model_params = self.validate_action_dict(action)

        # Create the appropriate PMF model
        if model_params["model_type"] == "single":
            # Use PMFSingle for scalar d2
            self._pmf_model = PMFSingle(
                d2=model_params["d2"],
                p_n=validated_params["p_n"],
                p_c=validated_params["p_c"],
                p_d=validated_params["p_d"],
                p_cd=validated_params["p_cd"],
                l_c=validated_params["l_c"],
                is_dot=validated_params["is_dot"],
                multiplier=validated_params["multiplier"],
            )
        else:  # model_params["model_type"] == "mixture"
            # Use PMFMixture for multiple d2 values
            self._pmf_model = PMFMixture(
                d2=model_params["d2"],
                w=model_params["w"],
                p_n=validated_params["p_n"],
                p_c=validated_params["p_c"],
                p_d=validated_params["p_d"],
                p_cd=validated_params["p_cd"],
                l_c=validated_params["l_c"],
                is_dot=validated_params["is_dot"],
                multiplier=validated_params["multiplier"],
            )

        # Get one-hit PMF from the underlying model
        self.one_hit_pmf = self._pmf_model.one_hit_pmf
        self.one_hit_support = self._pmf_model.one_hit_support

        # Number of hits
        self.n_hits = validated_params["n_hits"]

        # Initialize output arrays
        self.action_pmf = np.array([])
        self.action_support = np.array([])

        # Compute distributions and moments if requested
        if compute_distributions:
            self.action_pmf, self.action_support = self._compute_action_distribution(self.one_hit_pmf, self.one_hit_support)

        if compute_moments:
            self.action_pmf, self.action_support = self._compute_moments()

    @staticmethod
    def _coarsen_one_hit_pmf(one_hit_pmf: np.ndarray, action_delta: int) -> np.ndarray:
        """Coarsen a one-hit PMF by grouping nearby damage values.

        Parameters
        ----------
        one_hit_pmf : np.ndarray
            Probability mass function for one hit
        action_delta : int
            Step size for coarsening

        Returns
        -------
        np.ndarray
            Coarsened probability mass function with groups of damage values averaged
        """
        if action_delta == 1:
            return one_hit_pmf
        
        coarsened_length = one_hit_pmf.size // action_delta
        remainder = one_hit_pmf.size % action_delta
        remainder_weight = remainder / action_delta

        coarsened_one_hit_pmf = (
            one_hit_pmf[:-remainder].reshape((coarsened_length, -1)).mean(axis=1)
        )

        # Have the final index average the action_delta window
        # and any remainder, just ensure proper weighting.
        coarsened_one_hit_pmf[-1] *= 1 - remainder_weight
        coarsened_one_hit_pmf[-1] = remainder_weight * one_hit_pmf[-remainder:].mean()

        # Renormalize
        coarsened_one_hit_pmf *= action_delta

        return coarsened_one_hit_pmf

    @staticmethod
    def _coarsen_n_hit_support(
        one_hit_support: np.ndarray, n: int, delta: int
    ) -> np.ndarray:
        """Generate coarsened support for n-hit distribution.

        Parameters
        ----------
        one_hit_support : np.ndarray
            Support array for one-hit distribution
        n : int
            Number of hits
        delta : int
            Step size for coarsening

        Returns
        -------
        np.ndarray
            Coarsened support array for n-hit distribution with step size delta
        """
        coarsened_start = one_hit_support[0] * n

        # End is final index divisible by delta
        # (corresponds to the beginning of the final average window in the pmf).
        coarsened_remainder = one_hit_support.size % delta
        coarsened_end = one_hit_support[-(delta + coarsened_remainder)] * n

        coarsened_pmf = np.arange(coarsened_start, coarsened_end + delta, step=delta)

        return coarsened_pmf

    @staticmethod
    def _partition_n(n: int) -> tuple[int, int]:
        """Iteratively split integer n into partitions by dividing by 2.

        Parameters
        ----------
        n : int
            Number to partition

        Returns
        -------
        a : int
            First partition of n
        b : int
            Second partition of n, where a+b=n
        """
        if n % 2 == 0:
            a = n // 2
            b = n // 2

        else:
            a = (n + 1) // 2
            b = (n - 1) // 2

        return a, b

    @staticmethod
    def _convolve_by_partitions(one_hit_pmf, n: int) -> np.ndarray:
        """Self-convolve a 1-hit damage distribution to yield an n-hit damage distribution.

        This is efficiently performed by splitting n into partitions and convolving the 1-hit damage
        distribution by these partitions, significantly reducing the number of convolutions which
        needs to be performed.

        The partition method works as follows:
        1. If n is even, partition it into A=n/2 and B=n/2; otherwise, partition it into A=(n+1)/2 and B=(n-1)/2
        2. Re-partition B into A1 and B1
        3. Continue re-partitioning B until we reach 1

        Example: For n=13, this generates the partition set {1, 2, 3, 6, 7, 13}
        The convolution sequence would be:
        (1, 1) -> 2; (1, 2) -> 3; (3, 3) -> 6; (1, 6) -> 7; (6, 7) -> 13.

        Parameters
        ----------
        one_hit_pmf : np.ndarray
            Probability mass function of the damage distribution for one hit
        n : int
            Number of hits

        Returns
        -------
        np.ndarray
            Probability mass function for action landing n hits, normalized to sum to 1

        References
        ----------
        .. [1] https://math.stackexchange.com/questions/2114575/partitions-of-n-that-generate-all-numbers-smaller-than-n
        """
        a, b = ActionDistribution._partition_n(n)
        partition_set = set((a, b))
        while b > 1:
            a, b = ActionDistribution._partition_n(b)
            partition_set.update((a, b))

        # Happens if n = 1, just remove 0
        if 0 in partition_set:
            partition_set = partition_set.difference({0})

        # Also add n to the set for easy looping
        partition_set.update([n])
        partition_set = sorted(list(partition_set))

        # Now convolve according to the partition set
        # Keep track of results in a dictionary
        convolve_dict = {1: one_hit_pmf}

        # How to sum up the partitions to yield n sounds complicated, but there's only 3 cases
        for a in range(len(partition_set) - 1):
            # Self-add: e.g., 1 + 1 = 2
            if partition_set[a] + partition_set[a] == partition_set[a + 1]:
                convolve_dict[partition_set[a + 1]] = fftconvolve(
                    convolve_dict[partition_set[a]], convolve_dict[partition_set[a]]
                )

            # Add to previous partition: e.g., 7 + 6 = 13
            elif (a > 0) & (
                partition_set[a - 1] + partition_set[a] == partition_set[a + 1]
            ):
                convolve_dict[partition_set[a + 1]] = fftconvolve(
                    convolve_dict[partition_set[a - 1]],
                    convolve_dict[partition_set[a]],
                )

            # Add one: e.g., 6 + 1 = 7
            elif (a > 0) & (
                partition_set[a] + partition_set[0] == partition_set[a + 1]
            ):
                convolve_dict[partition_set[a + 1]] = fftconvolve(
                    convolve_dict[partition_set[0]], convolve_dict[partition_set[a]]
                )

            # Normalize, values can have numerical stability issues if
            # N > ~200
            convolve_dict[partition_set[a + 1]] /= np.trapezoid(
                convolve_dict[partition_set[a + 1]]
            )
        return convolve_dict[n]

    def _compute_action_distribution(
        self, one_hit_pmf: np.ndarray, one_hit_support: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the n-hit damage distribution for an action.

        Parameters
        ----------
        one_hit_pmf : np.ndarray
            Probability mass function for one hit
        one_hit_support : np.ndarray
            Support array representing possible damage values for one hit

        Returns
        -------
        coarsened_damage_distribution : np.ndarray
            The n-hit damage distribution PMF
        coarsened_support : np.ndarray
            The support array representing possible damage values
        """
        coarsened_one_hit_pmf = ActionDistribution._coarsen_one_hit_pmf(
            one_hit_pmf, self.action_delta
        )

        coarsened_support = ActionDistribution._coarsen_n_hit_support(
            one_hit_support, self.n_hits, self.action_delta
        )

        coarsened_damage_distribution = ActionDistribution._convolve_by_partitions(
            coarsened_one_hit_pmf, self.n_hits
        )
        return coarsened_damage_distribution, coarsened_support


class RotationDistribution:
    def __init__(self):
        """Distribution for a sequence of combat actions.

        Models the damage distribution of a full rotation consisting of multiple
        combat actions in sequence.
        """
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ffxiv_stats.moments import ActionMoments
    import pandas as pd

    p_n = 0.4
    p_c = 0.4
    p_d = 0.1
    p_cd = 0.1
    d2 = 10000
    l_c = 1500
    multiplier = 1.0

    action_dict = {
        "n": 10,
        "p_n": p_n,
        "p_c": p_c,
        "p_d": p_d,
        "p_cd": p_cd,
        "d2": d2,
        "l_c": 1500,
        "buffs": multiplier,
        "is_dot": False,
    }

    action_dict["d2"] = [10000, 12700]
    action_dict["w"] = [0.65, 0.35]
    action = pd.Series(
        {
            "n": 10,
            "p_n": p_n,
            "p_c": p_c,
            "p_d": p_d,
            "p_cd": p_cd,
            "d2": d2,
            "l_c": 1500,
            "buffs": multiplier,
            "is_dot": False,
        }
    )
    # am = ActionMoments(
    #     action,
    #     1,
    #     1,
    # )

    # PMFSingle(d2, p_n, p_c, p_d, p_cd, l_c, False, multiplier)
    ActionDistribution(action_dict, 1, 15, False, True)
    # PMFMixture([1000, 1272], [0.65, 0.35], p_n, p_c, p_d, p_cd, l_c, False, multiplier)
