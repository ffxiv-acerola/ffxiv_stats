import numpy as np

from ffxiv_stats.modifiers import level_mod


class Rate:
    def __init__(self, crit_amt, dh_amt, level=100) -> None:
        """
        Get probabilities of different hit types given critical hit and direct hit rate stats.
        """
        self.lvl_main = level_mod[level]["lvl_main"]
        self.lvl_sub = level_mod[level]["lvl_sub"]
        self.lvl_div = level_mod[level]["lvl_div"]

        self.crit_amt = crit_amt
        self.dh_amt = dh_amt
        self.p = self.get_p()
        self.l_c = self.crit_dmg_multiplier()
        pass

    def crit_dmg_multiplier(self) -> float:
        """
        Get the damage multiplier for landing a critical hit

        inputs:
        crit_amt: int, critical hit stat

        returns:
        critical hit damage multiplier
        """
        return np.floor(200 / self.lvl_div * (self.crit_amt - self.lvl_sub) + 1400)

    def crit_prob(self) -> float:
        """
        Get the probability of landing a critical hit

        inputs:
        crit_amt: int, critical hit stat

        returns:
        critical hit probability, from [0,1]
        """
        return np.floor(200 / self.lvl_div * (self.crit_amt - self.lvl_sub) + 50) / 1000

    def direct_hit_prob(self) -> float:
        """
        Get the probability of landing a direct hit

        inputs:
        crit_amt: int, direct hit stat

        returns:
        direct hit probability, from [0,1]
        """
        return np.floor(550 / self.lvl_div * (self.dh_amt - self.lvl_sub)) / 1000

    def get_p(self, ch_mod=0, dh_mod=0, guaranteed_hit_type=0):
        """
        Get the probability of each hit type occurring given the probability of a critical hit and direct hit

        inputs:
        crit_amt: Amount of critical hit stat.
        dh_amt: Amount of direct hit rate stat.
        ch_mod: Percentage to increase the base critical hit rate by if buffs are present.
                E.g., ch_mod=0.1 would add 0.1 to the base critical hit rate
        dh_mod: Percentage to increase the base direct hit rate by if buffs are present.
        guaranteed_hit_type: get probability for a guaranteed hit type.
                             Parameters ch_mod and dh_mod have no effect if this is non-zero
                             0 - no hit type guaranteed
                             1 - guaranteed critical hit
                             2 - guaranteed direct hit
                             3 - guaranteed critical-direct hit

        returns:
        probability of each hit type, [normal hit, critical hit given not CD hit, direct hit given not CDH hit, CDH hit]
        """
        # Floating point error can be resolved with round
        # Probabilities in FFXIV only use 3 significant digits because of flooring
        # The most number of significant digits is 6 for critical direct hits,
        # since it's a product of critical hit (3) and direct hit (3) = 6 sig digits
        # Just use 10 cause yolo, p sums to 1

        # Why does p have to sum to 1 without any floating point error?
        # scipy's multinomial weights will return nan if they do not.
        p_c = round(self.crit_prob() + ch_mod, 10)
        p_d = round(self.direct_hit_prob() + dh_mod, 10)

        if guaranteed_hit_type == 1:
            p_c = 1.0
        elif guaranteed_hit_type == 2:
            p_d = 1.0
        elif guaranteed_hit_type == 3:
            return np.array([0, 0, 0, 1.0])

        p_cd = round(p_c * p_d, 10)

        return np.array(
            [
                round(1.0 - p_c - p_d + p_cd, 10),
                round(p_c - p_cd, 10),
                round(p_d - p_cd, 10),
                p_cd,
            ]
        )

    def _guaranteed_critical_damage_multiplier(self, buff_crit_rate: float) -> float:
        """Damage bonus due to critical hit buffs present on a guaranteed critical hit.

        Args:
            buff_crit_rate (float): Amount the critical hit rate is buffed by. A 20% critical hit rate buff would be 0.20

        Returns:
            float: Damage multiplier
        """
        return round(1 + ((self.l_c / 1000 - 1) * buff_crit_rate), 3)

    def _guaranteed_direct_damage_multiplier(
        self, buff_dh_rate: float, determination: int, direct_hit_multiplier: float = 1.25
    ) -> float:
        """Damage bonus due to direct hit rate buffs present on a guaranteed direct hit.

        This is expressed in two components: (i) a direct hit rate component and (ii) determination component.

        The direct hit rate component depends on the magnitude of the direct hit rate buff.

        The determination component treats the direct hit rate stat as a bonus determination multiplier,
        
        f_det|adh = f_det + f_det(dh)

        Where f_det(dh) is computed using the formula for f_det but using the direct hit stat in lieu of determination. This will be present even if a guaranteed direct hit does not have a direct hit rate buff. The returned buff strength factors out the original 

        Args:
            buff_dh_rate (float): Amount the direct hit rate is buffed by. A 20% direct hit rate buff would be 0.2.
            direct_hit_multiplier (float, optional): Buff strength of a direct hit. Defaults to 1.25.

        Returns:
            float: Damage multiplier with new tenacity multiplier.
        """

        # Create a new determination multiplier with DH added, f_detDH
        dh_det_bonus = (
            1 + np.floor(140 * ((self.dh_amt - self.lvl_sub) / self.lvl_div)) / 1000
        )
        # If f_detDH is used as is, the determination multiplier would be double counted when potency is converted to damage.
        # Compute the normal f_det and divide it out.

        # Since Damage ~ potency * f_atk * f_det * buff_1 * ... * buff_n,
        # this is computing buff_i = f_detDH * buff_autoDH / f _det, so
        # Damage ~ potency * f_atk * f_detDH * buff_autoDH
        # Yeah, this can pry be done better.
        base_det = np.floor(140 * ((determination - self.lvl_sub) / self.lvl_div)) / 1000

        dh_bonus = 1 + buff_dh_rate * (direct_hit_multiplier - 1.0)
        return round((dh_det_bonus + base_det) / (1 + base_det) * dh_bonus, 3)

    def get_hit_type_damage_buff(
        self,
        guaranteed_hit_type=0,
        buff_crit_rate=0,
        buff_dh_rate=0,
        determination=None,
    ):
        """
        Compute the damage buff granted to a hit-type buff acting upon an action with a guaranteed hit type.
        guaranteed_hit_type: integer representing the guaranteed hit type:
                             0 - normal hit (why are you calling this function)
                             1 - critical hit
                             2 - direct hit
                             3 - critical direct hit
        buff_crit_rate: how much a buff increases the crit rate by, e.g., 10% increase -> 0.1.
                        Leave as 0 if no crit buff is present.
        buff_dh_rate: how much a buff increases the direct hit rate by, e.g., 10% increase -> 0.1.
                      Leave as 0 if no direct hit buff is present.
        determination: determination stat value. Used only for direct hit buffs,
                       as the direct hit rate stat gets added to the determination stat
                       to create and effective determination multiplier.
        """
        crit_buff = 1.0
        dh_buff = 1.0

        if guaranteed_hit_type == 0:
            return 1.0

        if guaranteed_hit_type in (1, 3):
            crit_buff = self._guaranteed_critical_damage_multiplier(buff_crit_rate)

        if guaranteed_hit_type in (2, 3):
            dh_buff = self._guaranteed_direct_damage_multiplier(buff_dh_rate, determination)

        return round(crit_buff * dh_buff, 6)


if __name__ == "__main__":
    pass
