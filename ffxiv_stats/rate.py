import numpy as np

from .modifiers import level_mod


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
        if guaranteed_hit_type == 0:
            return 1

        # Damage buff for crit rate buff
        unbuffed_crit_rate = self.crit_prob()
        buffed_crit_rate = round(unbuffed_crit_rate + buff_crit_rate, 3)

        crit_buff = 1 + (
            (buffed_crit_rate - unbuffed_crit_rate) * (unbuffed_crit_rate + 0.35)
        )

        if guaranteed_hit_type == 1:
            return crit_buff

        # Damage buff for direct hit rate buff
        unbuffed_dh_rate = self.direct_hit_prob()
        buffed_dh_rate = round(unbuffed_dh_rate + buff_dh_rate, 3)

        if determination is None:
            raise ValueError(
                "Determination stat value must be specified because Direct Hit rate buffs depend on the base Determination stat"
            )

        dh_buff = 1 + ((buffed_dh_rate - unbuffed_dh_rate) * 0.25)
        # Add DH rate to determination stat for new effective det multiplier
        det_and_dh_amt = (determination - self.lvl_main) + (self.dh_amt - self.lvl_sub)
        dh_buff *= np.floor(((140 * (det_and_dh_amt)) / self.lvl_div) + 1000) / 1000
        # Divide out the original determination multiplier so it isn't double counted.
        dh_buff /= (
            np.floor(((140 * (determination - self.lvl_main)) / self.lvl_div) + 1000)
        ) / 1000

        dh_buff = round(dh_buff, 6)
        # Works for either guaranteed direct hits or guaranteed critical-direct hits
        # for the former, crit_buff = 1
        return round(crit_buff * dh_buff, 6)


if __name__ == "__main__":
    pass
