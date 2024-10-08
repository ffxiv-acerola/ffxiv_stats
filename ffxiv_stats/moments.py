import pandas as pd
import numpy as np
from scipy.stats import multinomial
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def _coarsened_boundaries(start, end, delta):
    """
    Get the closest start and end points when coarsening a damage support.

    start - uncoarsened starting point
    end - uncoarsened ending point
    delta - step size
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


class Support:
    def __init__(self, d2, l_c, is_dot, buffs=None, l_d=None) -> None:
        """
        Class to compute the support of a single hit from an action.

        inputs:
        d2 - int, base damage of an action.
        l_c - int, modifier for landing a critical hit.
        is_dot - bool, whether the action is a DoT effect (has different support than direct dmg).
        buffs - list, List of any buffs present. A 10% damage buff would be `[1.10]`.
                If no buffs are present, then an empty list `[]`, list with none (`[None]`), or `[1]` can be supplied
        l_d - int, modifier for landing a direct hit. Not currently used, but might be for auto direct hit skills.
        """
        if l_d is None:
            self.l_d = 125

        self.d2 = d2
        self.l_c = l_c
        self.buff_prod = self.buff_check(buffs)
        self.is_dot = is_dot

        self.normal_supp = self.get_support("normal")
        self.crit_supp = self.get_support("critical")
        self.dir_supp = self.get_support("direct")
        self.crit_dir_supp = self.get_support("critical-direct")
        pass

    def buff_check(self, buffs):
        if buffs is None:
            return 1

        if isinstance(buffs, list) or isinstance(buffs, np.ndarray):
            if len(buffs) == 0:
                return 1
            else:
                return np.product(buffs)

        else:
            return buffs

    def ch_dmg_modifier(self):
        """
        Damage modifier for landing a critical hit.
        """
        return np.floor(np.floor(self.d2 * self.l_c) / 1000)

    def dh_dmg_modifier(self):
        """
        Damage modifier for landing a direct hit.
        """
        return np.floor(np.floor(self.d2 * self.l_d) / 100)

    def cdh_dmg_modifier(self):
        """
        Damage modifier for landing a critical-direct hit.
        """
        ch_damage = np.floor(np.floor(self.d2 * self.l_c) / 1000)
        return np.floor(np.floor(ch_damage * self.l_d) / 100)

    def get_support(self, hit_type):
        """
        Find the support (all possible damage values) of a single hit, given D2 and any buffs

        input:
        hit_type - str, type of hit. Can be `normal`, `critical`, `direct`, or `critical-direct`

        Returns:
        numpy array of the support for a single normal hit
        """
        if self.is_dot:
            lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2) + 1
            damage_range = np.arange(lower, upper)

            if hit_type == "normal":
                support = np.floor(damage_range * self.buff_prod)

            elif hit_type == "critical":
                support = np.floor(
                    np.floor(np.floor(damage_range * self.l_c) / 1000) * self.buff_prod
                )

            elif hit_type == "direct":
                support = np.floor(
                    np.floor(np.floor(damage_range * self.l_d) / 100) * self.buff_prod
                )

            elif hit_type == "critical-direct":
                ch_dmg = np.floor(np.floor(damage_range * self.l_c) / 1000)
                support = np.floor(
                    np.floor(np.floor(ch_dmg * self.l_d) / 100) * self.buff_prod
                )

            else:
                print("incorrect input")

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
            support = np.floor(damage_range * self.buff_prod)

        return support


class ActionMoments(Support):
    def __init__(
        self, action_df, t, action_delta=10, compute_mgf=True, moments_only=False
    ) -> None:
        """
        Compute moments for a action landing n_hits

        inputs:
        action_df - pandas dataframe with the columns:
                    n: int, number of hits.
                    p_n: probability of a normal hit.
                    p_c: probability of a critical hit.
                    p_d: probability of a direct hit.
                    p_cd: probability of a critical-direct hit.
                    d2: int, base damage value of action before any variability.
                    l_c: int, damage multiplier for a critical hit.
                              Value should be in the thousands (1250 -> 125% crit buff).
                    buffs: Total buff strength, or a list of buffs. A 10% buff should be represented as 1.1.
                           A 5% and 10% buff can be represented as either 1.155 or [1.05, 1.10], but the former is preferred.
                           Saving a dataframe with array columns can be finnicky.
                    is_dot: boolean or 0/1, whether the action is a damage over time effect.
        t - elapsed time, for converting damage to DPS
        action_delta - amount to discretize damage by. Instead of representing damage in steps of 1, 100, 101, 102, ..., 200,
                       damage is represented in steps of `action_delta`, 100, 110, 120, ..., 200.
                       Generally a value of 10 gives a good balance of speed and accuracy.
                       Larger values result in a faster calculation, but less accurate damage distributions.
        """

        column_check = set(["n", "d2", "l_c", "buffs", "is_dot"])
        if isinstance(action_df, pd.core.series.Series):
            supplied_columns = action_df.index
        else:
            supplied_columns = action_df.columns
        missing_columns = column_check - set(supplied_columns)
        if len(missing_columns) != 0:
            raise ValueError(
                f"The following column(s) are missing from `rotation_df`: {*missing_columns,}. Please refer to the docstring and add these field(s) or double check the spelling."
            )

        if any([x in ["p_n", "p_c", "p_d", "p_cd"] for x in list(supplied_columns)]):
            separated_p = True
        # Backwards compatibility when this was passed in as a list
        elif any([x in ["p"] for x in list(supplied_columns)]):
            separated_p = False
        else:
            raise ValueError(
                "No hit-type probability column detected. There should be four columns, p_n, p_c, p_d, and p_cd, for the probability of each hit type."
            )

        self.n = action_df["n"]

        if not separated_p:
            self.p = action_df["p"]
        else:
            self.p = np.array(
                [
                    action_df["p_n"],
                    action_df["p_c"],
                    action_df["p_d"],
                    action_df["p_cd"],
                ]
            )

        Support.__init__(
            self,
            action_df["d2"],
            action_df["l_c"],
            bool(action_df["is_dot"]),
            action_df["buffs"],
        )

        self.t = t
        self.compute_mgf = compute_mgf
        self.moments_only = moments_only

        if "action_name" in action_df:
            self.action_name = action_df["action_name"]

        # Compute first three moments, if specified
        if self.compute_mgf:
            # All possible hit types
            self.x = self.hit_type_combos(self.n)
            # Corresponding multinomial weight
            self.w = multinomial(self.n, self.p).pmf(self.x)

            # Lots of notation for computing moments when there are gaps
            self._S_N = self.normal_supp.size
            self._Z_N = np.sum(self.normal_supp)
            self._Z_N2 = np.sum(self.normal_supp**2)
            self._Z_N3 = np.sum(self.normal_supp**3)

            self._S_C = self.crit_supp.size
            self._Z_C = np.sum(self.crit_supp)
            self._Z_C2 = np.sum(self.crit_supp**2)
            self._Z_C3 = np.sum(self.crit_supp**3)

            self._S_D = self.dir_supp.size
            self._Z_D = np.sum(self.dir_supp)
            self._Z_D2 = np.sum(self.dir_supp**2)
            self._Z_D3 = np.sum(self.dir_supp**3)

            self._S_CD = self.crit_dir_supp.size
            self._Z_CD = np.sum(self.crit_dir_supp)
            self._Z_CD2 = np.sum(self.crit_dir_supp**2)
            self._Z_CD3 = np.sum(self.crit_dir_supp**3)

            self._first_moment = self._get_first_moment()
            self._second_moment = self._get_second_moment()
            self._third_moment = self._get_third_moment()

        if not self.moments_only:
            self.action_delta = action_delta
            self.damage_support, self.damage_distribution = (
                self.compute_dps_distribution()
            )
            self.dps_support = self.damage_support / self.t
            self.dps_distribution = self.damage_distribution / np.trapz(
                self.damage_distribution, self.dps_support
            )

        # Use MGF to compute moments if available, otherwise use
        self.mean = self.get_action_mean()
        self.variance = self.get_action_variance()
        self.skewness = self.get_action_skewness()
        self.standard_deviation = np.sqrt(self.variance)

        pass

    def hit_type_combos(self, n):
        """
        This will give all of the different hit combinations and will sum to n
        For example, if N = 10, some of the combinations are
        [0, 5, 5, 10]
        [10, 0, 0, 0]
        [3, 1, 3, 3]

        and so on

        Idk how it works because I copy/pasted from stackoverflow, but my God is it fast compared to nested loops
        https://stackoverflow.com/questions/34970848/find-all-combination-that-sum-to-n-with-multiple-lists
        """
        import itertools
        import operator

        hit_list = []
        for cuts in itertools.combinations_with_replacement(range(n + 1), 3):
            combi = list(map(operator.sub, cuts + (n,), (0,) + cuts))
            if max(combi) < n:
                hit_list.append(combi)

        # The loop doesn't include cases where there a n types of just 1 hit type
        # Hardcoding it in is easy
        hit_list.append([n, 0, 0, 0])
        hit_list.append([0, n, 0, 0])
        hit_list.append([0, 0, n, 0])
        hit_list.append([0, 0, 0, n])
        return np.array(hit_list)

    def _get_first_moment(self):
        """
        Compute the first moment (mean) for an action landing n hits.
        """
        first_deriv = (
            (self.x[:, 1] * self._Z_C) / self._S_C
            + (self.x[:, 3] * self._Z_CD) / self._S_CD
            + (self.x[:, 2] * self._Z_D) / self._S_D
            + (self.x[:, 0] * self._Z_N) / self._S_N
        )
        return np.dot(self.w, first_deriv)

    def _get_second_moment(self):
        """
        Compute the second moment for an action landing n hits
        """
        second_deriv = (
            (self.x[:, 1] ** 2 * self._Z_C**2) / self._S_C**2
            + (self.x[:, 3] ** 2 * self._Z_CD**2) / self._S_CD**2
            + self.x[:, 3]
            * (-(self._Z_CD**2 / self._S_CD**2) + self._Z_CD2 / self._S_CD)
            + (self.x[:, 2] ** 2 * self._Z_D**2) / self._S_D**2
            + self.x[:, 1]
            * (
                -(self._Z_C**2 / self._S_C**2)
                + self._Z_C2 / self._S_C
                + (2 * self.x[:, 3] * self._Z_C * self._Z_CD) / (self._S_C * self._S_CD)
                + (2 * self.x[:, 2] * self._Z_C * self._Z_D) / (self._S_C * self._S_D)
            )
            + self.x[:, 2]
            * (
                (2 * self.x[:, 3] * self._Z_CD * self._Z_D) / (self._S_CD * self._S_D)
                - self._Z_D**2 / self._S_D**2
                + self._Z_D2 / self._S_D
            )
            + (self.x[:, 0] ** 2 * self._Z_N**2) / self._S_N**2
            + self.x[:, 0]
            * (
                (2 * self.x[:, 1] * self._Z_C * self._Z_N) / (self._S_C * self._S_N)
                + (2 * self.x[:, 3] * self._Z_CD * self._Z_N) / (self._S_CD * self._S_N)
                + (2 * self.x[:, 2] * self._Z_D * self._Z_N) / (self._S_D * self._S_N)
                - self._Z_N**2 / self._S_N**2
                + self._Z_N2 / self._S_N
            )
        )
        return np.dot(self.w, second_deriv)

    def _get_third_moment(self):
        """
        Compute the third moment for an action landing n hits.
        """

        third_deriv = (
            (self.x[:, 1] ** 3 * self._Z_C**3) / self._S_C**3
            + (self.x[:, 3] ** 3 * self._Z_CD**3) / self._S_CD**3
            + self.x[:, 3] ** 2
            * (
                (-3 * self._Z_CD**3) / self._S_CD**3
                + (3 * self._Z_CD * self._Z_CD2) / self._S_CD**2
            )
            + self.x[:, 3]
            * (
                (2 * self._Z_CD**3) / self._S_CD**3
                - (3 * self._Z_CD * self._Z_CD2) / self._S_CD**2
                + self._Z_CD3 / self._S_CD
            )
            + (self.x[:, 2] ** 3 * self._Z_D**3) / self._S_D**3
            + self.x[:, 1] ** 2
            * (
                (-3 * self._Z_C**3) / self._S_C**3
                + (3 * self._Z_C * self._Z_C2) / self._S_C**2
                + (3 * self.x[:, 3] * self._Z_C**2 * self._Z_CD)
                / (self._S_C**2 * self._S_CD)
                + (3 * self.x[:, 2] * self._Z_C**2 * self._Z_D)
                / (self._S_C**2 * self._S_D)
            )
            + self.x[:, 2] ** 2
            * (
                (3 * self.x[:, 3] * self._Z_CD * self._Z_D**2)
                / (self._S_CD * self._S_D**2)
                - (3 * self._Z_D**3) / self._S_D**3
                + (3 * self._Z_D * self._Z_D2) / self._S_D**2
            )
            + self.x[:, 1]
            * (
                (2 * self._Z_C**3) / self._S_C**3
                - (3 * self._Z_C * self._Z_C2) / self._S_C**2
                + self._Z_C3 / self._S_C
                + (3 * self.x[:, 3] ** 2 * self._Z_C * self._Z_CD**2)
                / (self._S_C * self._S_CD**2)
                + self.x[:, 3]
                * (
                    (-3 * self._Z_C**2 * self._Z_CD) / (self._S_C**2 * self._S_CD)
                    + (3 * self._Z_C2 * self._Z_CD) / (self._S_C * self._S_CD)
                    - (3 * self._Z_C * self._Z_CD**2) / (self._S_C * self._S_CD**2)
                    + (3 * self._Z_C * self._Z_CD2) / (self._S_C * self._S_CD)
                )
                + (3 * self.x[:, 2] ** 2 * self._Z_C * self._Z_D**2)
                / (self._S_C * self._S_D**2)
                + self.x[:, 2]
                * (
                    (-3 * self._Z_C**2 * self._Z_D) / (self._S_C**2 * self._S_D)
                    + (3 * self._Z_C2 * self._Z_D) / (self._S_C * self._S_D)
                    + (6 * self.x[:, 3] * self._Z_C * self._Z_CD * self._Z_D)
                    / (self._S_C * self._S_CD * self._S_D)
                    - (3 * self._Z_C * self._Z_D**2) / (self._S_C * self._S_D**2)
                    + (3 * self._Z_C * self._Z_D2) / (self._S_C * self._S_D)
                )
            )
            + self.x[:, 2]
            * (
                (3 * self.x[:, 3] ** 2 * self._Z_CD**2 * self._Z_D)
                / (self._S_CD**2 * self._S_D)
                + (2 * self._Z_D**3) / self._S_D**3
                - (3 * self._Z_D * self._Z_D2) / self._S_D**2
                + self.x[:, 3]
                * (
                    (-3 * self._Z_CD**2 * self._Z_D) / (self._S_CD**2 * self._S_D)
                    + (3 * self._Z_CD2 * self._Z_D) / (self._S_CD * self._S_D)
                    - (3 * self._Z_CD * self._Z_D**2) / (self._S_CD * self._S_D**2)
                    + (3 * self._Z_CD * self._Z_D2) / (self._S_CD * self._S_D)
                )
                + self._Z_D3 / self._S_D
            )
            + (self.x[:, 0] ** 3 * self._Z_N**3) / self._S_N**3
            + self.x[:, 0] ** 2
            * (
                (3 * self.x[:, 1] * self._Z_C * self._Z_N**2)
                / (self._S_C * self._S_N**2)
                + (3 * self.x[:, 3] * self._Z_CD * self._Z_N**2)
                / (self._S_CD * self._S_N**2)
                + (3 * self.x[:, 2] * self._Z_D * self._Z_N**2)
                / (self._S_D * self._S_N**2)
                - (3 * self._Z_N**3) / self._S_N**3
                + (3 * self._Z_N * self._Z_N2) / self._S_N**2
            )
            + self.x[:, 0]
            * (
                (3 * self.x[:, 1] ** 2 * self._Z_C**2 * self._Z_N)
                / (self._S_C**2 * self._S_N)
                + (3 * self.x[:, 3] ** 2 * self._Z_CD**2 * self._Z_N)
                / (self._S_CD**2 * self._S_N)
                + (3 * self.x[:, 2] ** 2 * self._Z_D**2 * self._Z_N)
                / (self._S_D**2 * self._S_N)
                + (2 * self._Z_N**3) / self._S_N**3
                - (3 * self._Z_N * self._Z_N2) / self._S_N**2
                + self.x[:, 1]
                * (
                    (-3 * self._Z_C**2 * self._Z_N) / (self._S_C**2 * self._S_N)
                    + (3 * self._Z_C2 * self._Z_N) / (self._S_C * self._S_N)
                    + (6 * self.x[:, 3] * self._Z_C * self._Z_CD * self._Z_N)
                    / (self._S_C * self._S_CD * self._S_N)
                    + (6 * self.x[:, 2] * self._Z_C * self._Z_D * self._Z_N)
                    / (self._S_C * self._S_D * self._S_N)
                    - (3 * self._Z_C * self._Z_N**2) / (self._S_C * self._S_N**2)
                    + (3 * self._Z_C * self._Z_N2) / (self._S_C * self._S_N)
                )
                + self.x[:, 3]
                * (
                    (-3 * self._Z_CD**2 * self._Z_N) / (self._S_CD**2 * self._S_N)
                    + (3 * self._Z_CD2 * self._Z_N) / (self._S_CD * self._S_N)
                    - (3 * self._Z_CD * self._Z_N**2) / (self._S_CD * self._S_N**2)
                    + (3 * self._Z_CD * self._Z_N2) / (self._S_CD * self._S_N)
                )
                + self.x[:, 2]
                * (
                    (6 * self.x[:, 3] * self._Z_CD * self._Z_D * self._Z_N)
                    / (self._S_CD * self._S_D * self._S_N)
                    - (3 * self._Z_D**2 * self._Z_N) / (self._S_D**2 * self._S_N)
                    + (3 * self._Z_D2 * self._Z_N) / (self._S_D * self._S_N)
                    - (3 * self._Z_D * self._Z_N**2) / (self._S_D * self._S_N**2)
                    + (3 * self._Z_D * self._Z_N2) / (self._S_D * self._S_N)
                )
                + self._Z_N3 / self._S_N
            )
        )

        return np.dot(self.w, third_deriv)

    def get_action_mean(self):
        if self.compute_mgf:
            return self._first_moment / self.t
        else:
            return np.trapz(self.dps_support * self.dps_distribution, self.dps_support)

    def get_action_variance(self):
        if self.compute_mgf:
            return (self._second_moment - self.mean**2) / self.t**2
        else:
            return np.trapz(
                (self.dps_support - self.mean) ** 2 * self.dps_distribution,
                self.dps_support,
            )

    def get_action_skewness(self):
        if self.compute_mgf:
            return (
                self._third_moment - 3 * self.mean * self.variance - self.mean**3
            ) / self.variance ** (3.0 / 2.0)
        else:
            return np.trapz(
                ((self.dps_support - self.mean) / np.sqrt(self.variance)) ** 3
                * self.dps_distribution,
                self.dps_support,
            )

    def compute_dps_distribution(self):
        """
        Convolve the single-hit PMF of a action n_hit times to get the exact PMF of an action landing n_hits.

        returns:
        dmg_support - np array of the damage support
        conv_pmf - damage distribution for action landing n hits
        """

        def convolve_by_partitions(one_hit_pmf, n):
            """
            Self-convolve a 1-hit damage distribution to yield an n-hit damage distribution.
            This is efficiently performed by splitting n into partitions and convolving the 1-hit damage
            distribution by these partitions, significantly reducing the number of convolutions which
            needs to be performed.

            inputs:
            one_hit_pmf - np array of the damage distribution for one hit
            n - number of hits

            returns damage distribution for action landing n hits
            """

            def partition_n(n):
                """
                Iteratively split integer n into partitions by dividing by 2 until 1 is reached.
                """
                if n % 2 == 0:
                    a = n // 2
                    b = n // 2

                else:
                    a = (n + 1) // 2
                    b = (n - 1) // 2

                return a, b

            # The list of partitions are first computed by dividing by 2
            # (and adding or subtracting 1 for negative numbers)

            # Example: 13 -> {1, 2, 3, 6, 7, 13}
            # From the comment in https://math.stackexchange.com/questions/2114575/partitions-of-n-that-generate-all-numbers-smaller-than-n
            # I once studied this problem and found a constructive partition method. Here is the brief. We are given a positive integer n.
            # STEP ONE: if n is an even number, partition it into A=n2 and B=n2; otherwise, partition it into A=n+12 and B=n−12.
            # STEP TWO: re-partition B into A1 and B1.
            # STEP THREE: re-partition B1......Until we get 1.
            # I didn't prove this method always works but I believe it is valid
            a, b = partition_n(n)
            partition_set = set((a, b))
            while b > 1:
                a, b = partition_n(b)
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
                convolve_dict[partition_set[a + 1]] /= np.trapz(
                    convolve_dict[partition_set[a + 1]]
                )
            return convolve_dict[n]

        # Define the bounds of the mixture distribution (lowest roll NH and highest roll CDH)
        # Everything is integers, so the bounds can be defined with an arange
        min_roll = np.floor(self.normal_supp[0]).astype(int)
        max_roll = np.floor(self.crit_dir_supp[-1]).astype(int)

        # Need to find out how many indices away the start of each hit-type subdistribution is from
        # the lower bound of the mixture distribution.
        ch_offset = int(self.crit_supp[0] - self.normal_supp[0])
        dh_offset = int(self.dir_supp[0] - self.normal_supp[0])
        cdh_offset = int(self.crit_dir_supp[0] - self.normal_supp[0])

        # Set up slices to include gaps
        normal_slice = (self.normal_supp - self.normal_supp[0]).astype(int)
        ch_slice = (self.crit_supp - self.crit_supp[0] + ch_offset).astype(int)
        dh_slice = (self.dir_supp - self.dir_supp[0] + dh_offset).astype(int)
        cdh_slice = (self.crit_dir_supp - self.crit_dir_supp[0] + cdh_offset).astype(
            int
        )

        # Excluding gaps for now, this is causing a lot of issues.
        normal_range = (self.normal_supp - self.normal_supp[0]).astype(int)
        normal_slice = np.arange(normal_range.min(), normal_range.max() + 1, step=1)

        ch_range = (self.crit_supp - self.crit_supp[0] + ch_offset).astype(int)
        ch_slice = np.arange(ch_range.min(), ch_range.max() + 1, step=1)

        dh_range = (self.dir_supp - self.dir_supp[0] + dh_offset).astype(int)
        dh_slice = np.arange(dh_range.min(), dh_range.max() + 1, step=1)

        cdh_range = (self.crit_dir_supp - self.crit_dir_supp[0] + cdh_offset).astype(
            int
        )
        cdh_slice = np.arange(cdh_range.min(), cdh_range.max() + 1, step=1)
        # Mixture distribution defined with multinomial weights
        one_hit_pmf = np.zeros(max_roll - min_roll + 1)
        one_hit_pmf[normal_slice] = self.p[0] / self.normal_supp.size
        one_hit_pmf[ch_slice] = self.p[1] / self.crit_supp.size
        one_hit_pmf[dh_slice] = self.p[2] / self.dir_supp.size
        one_hit_pmf[cdh_slice] = self.p[3] / self.crit_dir_supp.size

        # The support needs to be able to account for trimming out
        # normal hits when there are guaranteed hit types.
        # Possible hit types are encoded as 1 (possible) and 0 (impossible)
        possible_hit_types = np.array([1, 1, 1, 1])
        possible_hit_types[self.p == 0] = 0

        # Lowest for each hit type
        # Guaranteed critical/direct hits make normal hits impossible
        lowest_roll = np.array(
            [
                self.normal_supp[0],
                self.crit_supp[0],
                self.dir_supp[0],
                self.crit_dir_supp[0],
            ]
        )
        # Lowest roll for only possible hit types
        lowest_roll = (possible_hit_types * lowest_roll)[
            (possible_hit_types * lowest_roll) > 0
        ].min() * self.n

        # If no DH is melded, it's impossible to DH
        highest_roll = np.array(
            [
                self.normal_supp[-1],
                self.crit_supp[-1],
                self.dir_supp[-1],
                self.crit_dir_supp[-1],
            ]
        )

        highest_roll = (possible_hit_types * highest_roll)[
            (possible_hit_types * highest_roll) > 0
        ].max() * self.n

        # 1-hit damage support
        one_hit_support = np.arange(
            lowest_roll // self.n, highest_roll // self.n + 1, step=1
        )

        # Coarsened 1-hit support
        coarsened_start, coarsened_end = _coarsened_boundaries(
            lowest_roll // self.n, highest_roll // self.n, self.action_delta
        )
        coarsened_one_hit_support = np.arange(
            coarsened_start, coarsened_end + self.action_delta, step=self.action_delta
        )
        # Coarsened with spacing of self.action_delta
        coarsened_n_hit_support = np.arange(
            coarsened_start * self.n,
            coarsened_end * self.n + self.action_delta,
            step=self.action_delta,
        ).astype(float)

        # If there are guaranteed hits, normal hits are impossible
        # and the one-hit pmf has a lot of zeros to the array.
        # This makes the convolution unnecessarily expensive.
        # These 0 values can be trimmed out
        one_hit_pmf = np.trim_zeros(one_hit_pmf)
        # Coarsen onto support with spacing `self.action_delta`
        coarsened_one_hit_pmf = np.interp(
            coarsened_one_hit_support, one_hit_support, one_hit_pmf
        )

        # Sometimes the gaps in the pmf can line up perfectly badly such that when the one-hit pmf
        # is coarsened, it interpolates to 0's. There should be exactly 5 unique probabilities,
        # Normal, critical, direct, critical-direct, and 0. If not, just ignore gaps in the support.
        # This technically isn't normalized, but the PMF is normalized at the end.
        if len(set(coarsened_one_hit_pmf)) != len(set(one_hit_pmf)):
            # Mixture distribution defined with multinomial weights
            one_hit_pmf = np.zeros(max_roll - min_roll + 1)
            one_hit_pmf[normal_slice[0] : normal_slice[-1]] = (
                self.p[0] / self.normal_supp.size
            )
            one_hit_pmf[ch_slice[0] : ch_slice[-1]] = self.p[1] / self.crit_supp.size
            one_hit_pmf[dh_slice[0] : dh_slice[-1]] = self.p[2] / self.dir_supp.size
            one_hit_pmf[cdh_slice[0] : cdh_slice[-1]] = (
                self.p[3] / self.crit_dir_supp.size
            )

            coarsened_one_hit_pmf = np.interp(
                coarsened_one_hit_support, one_hit_support, one_hit_pmf
            )

        conv_pmf = convolve_by_partitions(coarsened_one_hit_pmf, self.n)

        # Ensure distribution is normalized.
        return coarsened_n_hit_support, conv_pmf / np.trapz(
            conv_pmf, coarsened_n_hit_support
        )


class Rotation:
    def __init__(
        self,
        rotation_df,
        t,
        rotation_delta: int = 50,
        action_delta: int = 10,
        rotation_pdf_step: int = 0.5,
        action_pdf_step: int = 1,
        purge_action_moments=False,
        compute_mgf=True,
        convolve_all=False,
    ) -> None:
        """
        Get damage variability for a rotation.

        Inputs:
        rotation_df: rotation dataframe with the following columns and types:
                     action_name: str, unique name of an action. Unique action depends on `buffs`, `p`, and `l_c` present.
                     base_action: str, name of an action ignoring buffs. For example, Glare III with chain stratagem
                                       and Glare III with mug will have different `action_names`, but the same base_action.
                                       Used for grouping actions together.
                    p_n: probability of a normal hit.
                    p_c: probability of a critical hit.
                    p_d: probability of a direct hit.
                    p_cd: probability of a critical-direct hit.
                    d2: int, base damage value of action before any variability.
                    l_c: int, damage multiplier for a critical hit.
                              Value should be in the thousands (1250 -> 125% crit buff).
                     d2: int, base damage value of action before any variability.
                     l_c: int, damage multiplier for a critical hit.
                               Value should be in the thousands (1250 -> 125% crit buff).
                    buffs: Total buff strength, or a list of buffs. A 10% buff should be represented as 1.1.
                           A 5% and 10% buff can be represented as either 1.155 or [1.05, 1.10], but the former is preferred.
                           Saving a dataframe with array columns can be finnicky.
                     is_dot: boolean or 0/1, whether the action is a damage over time effect.
        t: float, time elapsed in seconds. Set t=1 to get damage dealt instead of DPS.
        convolve_all: bool, whether to compute all DPS distributions by convolutions (normally actions with large n can be computed with a skew normal distribution).
        action_delta - amount to discretize damage of actions by.
                       Instead of representing damage in steps of 1, 100, 101, 102, ..., 200,
                       damage is represented in steps of `action_delta`, 100, 110, 120, ..., 200.
                       Generally a value of 10 gives a good balance of speed and accuracy.
                       Larger values result in a faster calculation, but less accurate damage distributions.
        rotation_delta - Amount to discretize damage of unique actions by, for computing the rotation damage distribution.
                         Same rationale for actions, but just after all unique actions are grouped together.
        rotation_pdf_step - final step size used when reporting `self.rotation_dps_support/distribution`. Defaults to values of 0.5 DPS,
                            but can be changed to larger values if total damage dealt is being computed.
        action_pdf_step - final step size used when reported `self.action_dps_support/distributions` and `self.unique_actions_support/distribution`.
                          Defaults to a value of 1 but can be changed to larger values if total damage dealt is being computed.
        purge_action_moments - Keeping track of uncoarsened action distributions for full fight rotations can take up a moderate amount of memory /
                               disk space when pickled due to array size. "Purging" these removes them  to reduce resource requirements.
                               These are largely intermediate variables for calculating # more familiar values - people are generally interested
                               in the DPS distribution for all broil IV  casts, not the individual DPS distributions for Broil IV,
                               Broil IV with Chain and Tech, and Broil IV with tech but not chain.
        """
        column_check = set(["base_action", "action_name"])
        missing_columns = column_check - set(rotation_df.columns)
        if len(missing_columns) != 0:
            raise ValueError(
                f"The following column(s) are missing from `rotation_df`: {*missing_columns,}. Please refer to the docstring and add these field(s) or double check the spelling."
            )

        self.rotation_df = rotation_df
        self.t = t
        # Deprecated/currently unused
        self.convolve_all = convolve_all
        # Damage is discretized by this much.
        # Bigger number = faster but larger discretization error
        # Smaller number = slower but more accurate.
        self.rotation_delta = rotation_delta
        self.action_delta = action_delta

        # When damage distributions are saved, have the spacing be this much
        # Useful for changing spacing with damage (t=1) vs DPS
        # DPS steps of 0.5 - 10 make sense
        # Damage, steps of 100 - 1000 make sense.
        self.rotation_pdf_step = rotation_pdf_step
        self.action_pdf_step = action_pdf_step
        self.purge_action_moments = purge_action_moments
        self.compute_mgf = compute_mgf

        self.action_moments = [
            ActionMoments(row, t, action_delta=action_delta, compute_mgf=compute_mgf)
            for _, row in rotation_df.iterrows()
        ]
        self.action_names = rotation_df["action_name"].tolist()
        self.action_means = np.array([x.mean for x in self.action_moments])
        self.action_variances = np.array([x.variance for x in self.action_moments])
        self.action_std = np.sqrt(self.action_variances)
        self.action_skewness = np.array([x.skewness for x in self.action_moments])

        self.rotation_mean = np.sum(self.action_means)
        self.rotation_variance = np.sum(self.action_variances)
        self.rotation_std = np.sqrt(self.rotation_variance)
        # Need just the numerator of Pearson's skewness, which is why we multiply by the action variances inside the sum
        self.rotation_skewness = np.sum(
            self.action_skewness * self.action_variances ** (3 / 2)
        ) / np.sum(self.action_variances) ** (3 / 2)

        self.compute_dps_distributions()

        if self.purge_action_moments:
            self.action_moments = [None] * len(self.action_moments)
            del self.action_dps_distributions
            del self.action_dps_support
        pass

    def compute_dps_distributions(self) -> None:
        """
        Compute and set the support and PMF of DPS distributions.

        This method is broken into 2 sections
        (i) Unique actions (Action A with Buff 1 and Action A with Buff 2 are group together now).
        (ii) The entire rotation.
        Specifics on convolving everything together because there are quite a few nuances to
        do things efficiently while still being correct.
        All damage distributions is convolved together using damage and not DPS.
        The supports of each distribution must be on the same grid for the convolution to correctly
        correspond to a sum of random variables. Converting to DPS usually ends up with floats, so
        dealing with integer values of damage is a much more convenient unit to work in.

        At first, we just keep everything in terms of damage, and the supports are just
        all integers from the lower to upper bound. However, this makes the convolutions very expensive.
        The computational cost is N log N, where N is the number of integers between the lower (all hits normal)
        and upper (all hits critical-direct) bound. This can get very large (N ~ 1e7-1e8) and become
        computationally expensive, even with N log N complexity. Instead of working in steps of 1 damage,
        we can work in higher steps of damage, like 100/1000/10000/etc, by interpolating the damage
        distributions to a coarser grid. This process is referred to as "coarsening".

        The major consideration for coarsening is when to coarsen and by how much.
        Coarsening leads to a greater reduction in computational efficiency when N becomes large.
        All action distributions are initially convolved in steps of 1 damage n_hit times.
        Unique action distributions are also convolved in steps of 1 damage, and then coarsened.

        The action with the smallest damage span will limit how much the support can be coarsened by.
        The auto-attacks of a WHM only span 10s of damage, but their Afflatus Misery action can span 10,000s.
        This is a somewhat unique case, which also makes the argument that auto-attacks can be ignored.
        By default, damage is discretized in steps of 250, which seemed to still give good accuracy.
        This also means that actions with very low damage spans are ignored, like healer auto attacks.
        This wont have a large impact on the rotation damage distribution.
        A future update might work on dynamically setting this value, or allow for different spacings,
        which are unified at the very end.
        """

        # section (0), individual actions, just unpack from the action moments
        self.action_dps_support = [x.damage_support for x in self.action_moments]
        self.action_dps_distributions = [
            x.damage_distribution for x in self.action_moments
        ]
        self.rotation_dps_distribution = None

        # Section (ii) base actions
        # Neat little function which says which base_action each index belongs to
        idx, unique_action_names = pd.factorize(self.rotation_df["base_action"])
        self.unique_actions = {}
        self.unique_actions = {n: [] for n in unique_action_names}

        for i, x in enumerate(idx):
            self.unique_actions[unique_action_names[x]].append(i)

        self.unique_actions_distribution = {}

        # Now loop over unique action indices and convolve together
        for _, (name, action_idx_list) in enumerate(self.unique_actions.items()):
            action_low_high = np.zeros((len(self.unique_actions[name]), 2))

            # Support is sum of all lowest possible value (min roll NH) to highest possible value (max roll CDH)
            for idx, action_idx in enumerate(action_idx_list):
                action_low_high[idx, :] = np.array(
                    [
                        self.action_dps_support[action_idx].min(),
                        self.action_dps_support[action_idx].max(),
                    ]
                )

            if len(action_idx_list) == 1:
                action_dps_distribution = self.action_dps_distributions[
                    action_idx_list[0]
                ]

            elif len(action_idx_list) > 1:
                action_dps_distribution = fftconvolve(
                    self.action_dps_distributions[action_idx_list[0]],
                    self.action_dps_distributions[action_idx_list[1]],
                )

            if len(action_idx_list) > 2:
                for idx in range(1, len(action_idx_list) - 1):
                    action_dps_distribution = fftconvolve(
                        action_dps_distribution,
                        self.action_dps_distributions[action_idx_list[idx + 1]],
                    )
                # For some reason these numbers get super tiny fast
                # and can lead to underflow, so periodically make them on the order of 1
                # if they get too small. Normalization is irrelevant since these get normalized
                # later
                if action_dps_distribution.max() < 1e-75:
                    action_dps_distribution /= action_dps_distribution.max()

            # Coarsen support in prep for rotation distribution
            uncoarsened_support = np.arange(
                action_low_high[:, 0].sum(),
                action_low_high[:, 1].sum() + self.action_delta,
                step=self.action_delta,
            )

            coarsened_start, coarsened_end = _coarsened_boundaries(
                action_low_high[:, 0].sum(),
                action_low_high[:, 1].sum(),
                self.rotation_delta,
            )

            coarsened_support = np.arange(
                coarsened_start,
                coarsened_end + self.rotation_delta,
                step=self.rotation_delta,
            )

            action_dps_distribution = np.interp(
                coarsened_support, uncoarsened_support, action_dps_distribution
            )

            self.unique_actions_distribution[name] = {
                "support": coarsened_support,
                "dps_distribution": action_dps_distribution,
            }

        # Section (iii) whole rotation
        rotation_lower_bound = np.array(
            [v["support"][0] for _, v in self.unique_actions_distribution.items()]
        ).sum()
        rotation_upper_bound = np.array(
            [v["support"][-1] for _, v in self.unique_actions_distribution.items()]
        ).sum()

        # `self.rotation_dps_distribution` needs to first be defined by convolving the first two unique actions together
        # then we can loop starting at the second index.
        if len(unique_action_names) > 1:
            self.rotation_dps_distribution = fftconvolve(
                self.unique_actions_distribution[unique_action_names[0]][
                    "dps_distribution"
                ],
                self.unique_actions_distribution[unique_action_names[1]][
                    "dps_distribution"
                ],
            )
        # Special case if theres only one action, just return the first element.
        else:
            self.rotation_dps_distribution = self.unique_actions_distribution[
                unique_action_names[0]
            ]["dps_distribution"]

        # Now loop
        if len(unique_action_names) > 2:
            for a in range(2, len(unique_action_names)):
                self.rotation_dps_distribution = fftconvolve(
                    self.unique_actions_distribution[unique_action_names[a]][
                        "dps_distribution"
                    ],
                    self.rotation_dps_distribution,
                )
                if self.rotation_dps_distribution.max() < 1e-75:
                    self.rotation_dps_distribution /= (
                        self.rotation_dps_distribution.max()
                    )

        # Create support and convert to DPS
        # Boundaries for coarsened distribution
        coarsened_rotation_start, coarsened_rotation_end = _coarsened_boundaries(
            rotation_lower_bound,
            rotation_upper_bound,
            self.rotation_delta,
        )
        rotation_dps_support = (
            np.arange(
                coarsened_rotation_start,
                coarsened_rotation_end + self.rotation_delta,
                step=self.rotation_delta,
            ).astype(float)
            / self.t
        )

        self.rotation_dps_support = np.arange(
            int(rotation_dps_support[1]),
            int(rotation_dps_support[-1]) + self.rotation_pdf_step,
            step=self.rotation_pdf_step,
        )
        self.rotation_dps_distribution = np.interp(
            self.rotation_dps_support,
            rotation_dps_support,
            self.rotation_dps_distribution,
        )
        # And renormalize the DPS distribution
        self.rotation_dps_distribution /= np.trapz(
            self.rotation_dps_distribution, self.rotation_dps_support
        )

        # Now all the damage distributions have been computed, can convert to DPS
        # action dps distributions. We also coarsen the support to 0.5 DPS
        # or else this uses a lot of memory
        for idx in range(len(self.action_dps_distributions)):
            lower, upper = (
                self.action_dps_support[idx][0] / self.t,
                self.action_dps_support[idx][-1] / self.t,
            )
            # Some actions like healer autos don't span a large DPS range and don't need to be coarsened.
            if upper - lower > 10:
                new_action_support = np.arange(
                    int(lower),
                    int(upper) + self.action_pdf_step,
                    step=self.action_pdf_step,
                )
                self.action_dps_distributions[idx] = np.interp(
                    new_action_support,
                    self.action_dps_support[idx] / self.t,
                    self.action_dps_distributions[idx],
                )
                self.action_dps_support[idx] = new_action_support

            self.action_dps_distributions[idx] /= np.trapz(
                self.action_dps_distributions[idx], self.action_dps_support[idx]
            )

        for u in unique_action_names:
            self.unique_actions_distribution[u]["support"] /= self.t
            self.unique_actions_distribution[u]["dps_distribution"] /= np.trapz(
                self.unique_actions_distribution[u]["dps_distribution"],
                self.unique_actions_distribution[u]["support"],
            )

        pass

    @classmethod
    def moments_to_skew_norm(self, mean, variance, skewness):
        """
        Converts the mean, variance, and Pearson's skewness to parameters defined by skew normal.
        The parameters are not the same, but can be interconverted: https://en.wikipedia.org/wiki/Skew_normal_distribution
        """

        delta = np.sqrt(
            np.pi
            / 2
            * (np.abs(skewness)) ** (2 / 3)
            / (np.abs(skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        alpha = np.sign(skewness) * delta / np.sqrt(1 - delta**2)
        omega = np.sqrt(variance / (1 - 2 * delta**2 / np.pi))
        squigma = mean - omega * delta * np.sqrt(2 / np.pi)

        return alpha, omega, squigma

    def plot_action_distributions(self, ax=None, **kwargs):
        """
        Plot DPS distributions for each unique action.
        Recall a action is unique based on the action *and* the buffs present.
        Action A with Buff 1 is different than Action A with Buff 2.

        inputs:
        ax - matplotlib axis object, optional axis object to plot onto.
             If one is not supplied, a new figure and shown at the end.
        **kwargs - any kwargs to be passed to, `ax.plot(**kwargs)`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        for a in range(self.action_means.size):
            ax.plot(
                self.action_dps_support[a],
                self.action_dps_distributions[a],
                label=self.action_names[a],
            )
        ax.set_xlabel("Damage per Second (DPS)")

        if return_ax:
            return ax
        else:
            plt.show()
            pass

    def plot_unique_action_distributions(self, ax=None, **kwargs):
        """
        Plot DPS distribution for unique actions, grouped by action name.
        For example, this would should the sum of DPS distributions for Action A with Buff 1 and Action A with Buff 2
        and label it as "Action A".

        inputs:
        ax - matplotlib axis object, optional axis object to plot onto.
             If one is not supplied, a new figure and shown at the end.
        **kwargs - any kwargs to be passed to, `ax.plot(**kwargs)`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        for _, (name, distributions) in enumerate(
            self.unique_actions_distribution.items()
        ):
            ax.plot(
                distributions["support"],
                distributions["dps_distribution"],
                label=name,
                **kwargs,
            )

        ax.legend()
        if return_ax:
            return ax
        else:
            plt.show()
            pass

    def plot_rotation_distribution(self, ax=None, **kwargs):
        """
        Plot the overall DPS distribution for the rotation.

        inputs:
        ax - matplotlib axis object, optional axis object to plot onto.
             If one is not supplied, a new figure and shown at the end.
        **kwargs - any kwargs to be passed to, `ax.plot(**kwargs)`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        ax.plot(self.rotation_dps_support, self.rotation_dps_distribution, **kwargs)
        ax.set_xlabel("Damage per Second (DPS)")

        if return_ax:
            return ax
        else:
            plt.show()
            pass


if __name__ == "__main__":
    pass
