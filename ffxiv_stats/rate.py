import numpy as np
from .modifiers import level_mod


class Rate:
    def __init__(self, crit_amt, dh_amt, level=90) -> None:
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
            p_c = 1.
        elif guaranteed_hit_type == 2:
            p_d = 1.
        elif guaranteed_hit_type == 3:
            return np.array([0, 0, 0, 1.])
        
        p_cd = round(p_c * p_d, 10)

        return np.array(
            [
                round(1.0 - p_c - p_d + p_cd, 10),
                round(p_c - p_cd, 10),
                round(p_d - p_cd, 10),
                p_cd,
            ]
        )
