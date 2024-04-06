import numpy as np
from numpy import floor as nf

from warnings import warn

from .moments import Rotation
from .modifiers import level_mod


class BaseStats(Rotation):
    def __init__(
        self,
        attack_power: int,
        trait: int,
        main_stat: int,
        det: int,
        crit_stat: int,
        dh_stat: int,
        dot_speed_stat: int,
        auto_speed_stat: int,
        weapon_damage: int,
        delay,
        strength: int = None,
        tenacity: int = 400,
        pet_attack_power: int = None,
        pet_attack_power_scalar: int = None,
        pet_attack_power_offset: int = None,
        pet_job_attribute: int = 100,
        pet_atk_mod: int = 195,
        level: int = 90,
    ) -> None:
        """
        Base class for converting potency to base damage dealt. Not meant to be used alone.
        Instead use a `ROLE` class, which inherits this class and applies/sets the correct attributes and stats.
        """
        # Level dependent parameters
        # currently for lvl 90
        self.lvl_main = level_mod[level]["lvl_main"]
        self.lvl_sub = level_mod[level]["lvl_sub"]
        self.lvl_div = level_mod[level]["lvl_div"]
        self.job_attribute = 115
        self.atk_mod = 195

        self.main_stat = main_stat
        self.strength = strength

        self.weapon_damage = weapon_damage
        self.attack_power = attack_power

        self.det = det
        self.tenacity = tenacity
        self.crit_stat = crit_stat
        self.dh_stat = dh_stat

        self.trait = trait
        self.auto_trait = trait

        self.dot_speed_stat = dot_speed_stat

        self.auto_speed_stat = auto_speed_stat
        self.delay = delay

        # Pet attributes, if not specified, pet methods will not work
        # Python formatter makes this a tuple, IDK why.
        self.pet_job_attribute = pet_job_attribute
        self.pet_attack_power = pet_attack_power
        self.pet_attack_power_scalar = pet_attack_power_scalar
        self.pet_attack_power_offset = pet_attack_power_offset

        # Apply pet scalar and offset for an effective pet attack power
        self.pet_effective_attack_power = nf(
            self.pet_attack_power_scalar
            * (self.pet_attack_power + self.pet_attack_power_offset)
        )

        self.pet_atk_mod = pet_atk_mod

        self.attack_multiplier = self.f_atk()
        self.determination_multiplier = self.f_det()
        self.tenacity_multiplier = self.f_ten()
        self.weapon_damage_multiplier = self.f_wd()
        pass

    def attach_rotation(self, rotation_df, t, action_delta=10, rotation_delta=100):
        """
        Attach a rotation data frame and compute the corresponding DPS distribution.

        Inputs
        rotation_df - pandas dataframe, dataframe containing rotation attributes. Should have the following columns:
                      action_name: str, unique name of an action. Unique action depends on `buffs`, `p`, and `l_c` present.
                      base_action: str, name of an action ignoring buffs. For example, Glare III with chain stratagem
                                        and Glare III with mug will have different `action_names`, but the same base_action.
                                        Used for grouping actions together.
                      potency: int, potency of the action
                      n: int, number of hits for the action.
                      p_n: probability of a normal hit.
                      p_c: probability of a critical hit.
                      p_d: probability of a direct hit.
                      p_cd: probability of a critical-direct hit.
                      l_c: int, damage multiplier for a critical hit.
                                Value should be in the thousands (1250 -> 125% crit buff).
                      buffs: Total buff strength, or a list of buffs. A 10% buff should be represented as 1.1.
                             A 5% and 10% buff can be represented as either 1.155 or [1.05, 1.10], but the former is preferred.
                             Saving a dataframe with array columns can be finnicky.
                      damage_type: str saying the type of damage, {'direct', 'magic-dot', 'physical-dot', 'auto'}
                      main_stat_add: int, how much to add to the main stat (used to account for medication, if present) when computing d2
        t - time elapsed for computing DPS from damage.
        action_delta - amount to discretize damage of actions by.
                       Instead of representing damage in steps of 1, 100, 101, 102, ..., 200,
                       damage is represented in steps of `action_delta`, 100, 110, 120, ..., 200.
                       Generally a value of 10 gives a good balance of speed and accuracy.
                       Larger values result in a faster calculation, but less accurate damage distributions.
        rotation_delta - Amount to discretize damage of unique actions by, for computing the rotation damage distribution.
                         Same rationale for actions, but just after all unique actions are grouped together.
        """
        column_check = set(["potency", "damage_type"])
        missing_columns = column_check - set(rotation_df.columns)
        if len(missing_columns) != 0:
            raise ValueError(
                f"The following column(s) are missing from `rotation_df`: {*missing_columns,}. Please refer to the docstring and add these field(s) or double check the spelling."
            )

        d2 = []
        is_dot = []
        for _, row in rotation_df.iterrows():
            if row["damage_type"] == "direct":
                d2.append(
                    self.direct_d2(row["potency"], ap_adjust=row["main_stat_add"])
                )
                is_dot.append(0)

            elif row["damage_type"] == "magic-dot":
                d2.append(
                    self.dot_d2(
                        row["potency"], magic=True, ap_adjust=row["main_stat_add"]
                    )
                )
                is_dot.append(1)

            elif row["damage_type"] == "physical-dot":
                d2.append(
                    self.dot_d2(
                        row["potency"], magic=False, ap_adjust=row["main_stat_add"]
                    )
                )
                is_dot.append(1)

            elif row["damage_type"] == "auto":
                # Medication doesn't affect healer autos
                # but it affects all others
                if isinstance(self, Healer):
                    ap_adjust = 0
                else:
                    ap_adjust = row["main_stat_add"]

                d2.append(self.auto_attack_d2(row["potency"], ap_adjust=ap_adjust))
                is_dot.append(0)

            elif row["damage_type"] == "pet":
                d2.append(
                    self.pet_direct_d2(row["potency"], ap_adjust=row["main_stat_add"])
                )
                is_dot.append(0)
            else:
                raise ValueError(
                    f"Invalid damage type value of '{row['damage_type']}'. Allow values are ('direct', 'magic-dot', 'physical-dot', 'auto')"
                )

        rotation_df["d2"] = d2
        rotation_df["is_dot"] = is_dot

        super().__init__(
            rotation_df, t, action_delta=action_delta, rotation_delta=rotation_delta
        )
        pass

    def add_role(self, role):
        """
        Add a role attribute to the object.

        inputs:
        role - str, which role is being modeled.
        """
        self.role = role
        pass

    def add_job_name(self, job_name):
        """
        Add a job attribute to the object.

        inputs:
        role - str, which job is being modeled.
        """
        self.job = job_name
        pass

    def add_description(self, description):
        """
        Add a description to the object. For example this could be a specific build, rotation, etc.

        inputs:
        role - str, description of this object.
        """
        self.description = description
        pass

    def f_wd(self):
        """
        Calculate weapon damage multiplier.
        """
        return np.floor(
            (self.lvl_main * self.job_attribute / 1000) + self.weapon_damage
        )

    def f_atk(self, ap_adjust=0):
        """
        Calculate attack multiplier.

        Inputs
        ap_adjust - int, additional amount to add to attack power (main stat). Used to account for medication.
        """
        return (
            np.floor(
                self.atk_mod
                * ((self.attack_power + ap_adjust) - self.lvl_main)
                / self.lvl_main
            )
            + 100
        )

    def f_det(self):
        """
        Calculate determination multiplier.
        """
        return np.floor(140 * (self.det - self.lvl_main) / self.lvl_div + 1000)

    def f_ten(self):
        """
        Calculate tenacity damage multiplier.
        """
        return np.floor(100 * (self.tenacity - self.lvl_sub) / self.lvl_div + 1000)

    def f_speed_dot(self):
        """
        Calculate speed multiplier for damage over time attacks.
        """
        return np.floor(
            130 * (self.dot_speed_stat - self.lvl_sub) / self.lvl_div + 1000
        )

    def f_speed_auto(self):
        """
        Calculate speed multiplier for auto attacks.
        """
        return np.floor(
            130 * (self.auto_speed_stat - self.lvl_sub) / self.lvl_div + 1000
        )

    def f_auto(self):
        return np.floor(self.f_wd() * self.delay / 3)

    def get_gcd(self):
        # TODO: add GCD (probably not essential?)
        pass

    def auto_attack_d2(self, potency, ap_adjust=0, stat_override=None):
        """
        Get base damage of an auto-attack before any variability.

        inputs:
        potency - int, potency of an attack
        ap_adjust - int, amount of main stat to add. Used to account for medication.
        """

        # Account for healer auto attacks.
        # who use strength for AA but main stat is mind
        # All other jobs have AA scale off of main stat
        if isinstance(self, Healer) or isinstance(self, MagicalRanged):
            atk = (
                np.floor(
                    self.atk_mod
                    * ((self.strength + ap_adjust) - self.lvl_main)
                    / self.lvl_main
                )
                + 100
            )
        else:
            atk = self.f_atk(ap_adjust)

        auto_d1 = nf(nf(nf(potency * atk * self.f_det()) / 100) / 1000)
        auto_d2 = nf(
            nf(
                nf(
                    nf(
                        nf(
                            nf(
                                nf(nf(auto_d1 * self.f_ten()) / 1000)
                                * self.f_speed_auto()
                            )
                            / 1000
                        )
                        * self.f_auto()
                    )
                    / 100
                )
                * self.trait
            )
            / 100
        )
        return auto_d2

    def direct_d2(self, potency, ap_adjust=0):
        """
        Get base damage of direct damage before any variability.
        Can be called directly or is automatically called b y the `attach_rotation` method.

        inputs:
        potency - int, potency of an attack
        ap_adjust - int, amount of main stat to add. Used to account for medication.
        """
        d1 = nf(nf(nf(potency * self.f_atk(ap_adjust) * self.f_det()) / 100) / 1000)

        return nf(
            nf(
                nf(nf(nf(nf(d1 * self.f_ten()) / 1000) * self.f_wd()) / 100)
                * self.trait
            )
            / 100
        )

    def dot_d2(self, potency, magic=True, ap_adjust=0):
        if magic:
            dot_d1 = nf(
                nf(
                    nf(
                        nf(nf(nf(potency * self.f_wd()) / 100) * self.f_atk(ap_adjust))
                        / 100
                    )
                    * self.f_speed_dot()
                )
                / 1000
            )
            return (
                nf(
                    nf(
                        nf(
                            nf(nf(nf(dot_d1 * self.f_det()) / 1000) * self.f_ten())
                            / 1000
                        )
                        * self.trait
                    )
                    / 100
                )
                + 1
            )
        else:
            dot_d1 = nf(nf(nf(potency * self.f_atk() * self.f_det()) / 100) / 1000)
            return (
                nf(
                    nf(
                        nf(
                            nf(
                                nf(
                                    nf(
                                        nf(nf(dot_d1 * self.f_ten()) / 1000)
                                        * self.f_speed_dot()
                                    )
                                    / 1000
                                )
                                * self.f_wd()
                            )
                            / 100
                        )
                        * self.trait
                    )
                    / 100
                )
                + 1
            )

    @staticmethod
    def undo_main_stat_party_bonus(percent_bonus, main_stat_with_bonus):
        """
        Estimate how much main stat is applied by the party bonus.
        Used for subtracting out for pet potency.
        It is an estimate because of floor rounding, but should be within 1 point.
        """
        undone_stat_float = main_stat_with_bonus / percent_bonus

        # Try to account for integer math by taking the floor and ceiling and seeing which one leads to the correct party bonus value
        floored_undone_stat = int(np.floor(undone_stat_float))
        ceilinged_undone_stat = int(np.ceil(undone_stat_float))

        if np.floor(floored_undone_stat * percent_bonus) == main_stat_with_bonus:
            return floored_undone_stat

        else:
            return ceilinged_undone_stat

    pass

    def pet_f_atk(self, ap_adjust=0):
        """
        Calculate attack multiplier.

        Inputs
        ap_adjust - int, additional amount to add to attack power (main stat). Used to account for medication.
        """
        return (
            np.floor(
                self.pet_atk_mod
                * ((self.pet_effective_attack_power + ap_adjust) - self.lvl_main)
                / self.lvl_main
            )
            + 100
        )

    def pet_f_wd(self):
        """
        Calculate weapon damage multiplier.
        """
        return np.floor(
            (self.lvl_main * self.pet_job_attribute / 1000) + self.weapon_damage
        )

    def pet_direct_d2(self, potency, ap_adjust=0):
        """
        Get base damage of direct damage before any variability.
        Can be called directly or is automatically called by the `attach_rotation` method.

        inputs:
        potency - int, potency of an attack
        ap_adjust - int, amount of main stat to add. Used to account for medication.
        """
        d1 = nf(nf(nf(potency * self.pet_f_atk(ap_adjust) * self.f_det()) / 100) / 1000)

        return nf(nf(nf(nf(nf(d1 * self.f_ten()) / 1000) * self.pet_f_wd()) / 100) * self.trait / 100)


class Healer(BaseStats):
    def __init__(
        self,
        mind: int,
        strength: int,
        det: int,
        spell_speed: int,
        crit_stat: int,
        dh_stat: int,
        weapon_damage: int,
        delay: float,
        pet_attack_power: int = None,
        pet_attack_power_scalar: float = 1.058,
        pet_attack_power_offset: int = 0,
        pet_job_attribute: int = 100,
        pet_atk_mod: int = 195,
        level: int = 90,
        intelligence=None,
        dexterity=None,
        vit=None,
        tenacity=None,
    ) -> None:
        """Set Healer stats to compute damage from potency.

        Many of the pet stats are used to compute an effective pet attack power, given by:
        effective_pet_attack_power = floor(pet_attack_power_scalar * (pet_attack_power + pet_attack_power_offset))

        Args:
            mind (int): Mind
            strength (int): Strength stat, for auto attacks.
            det (int): Determination stat.
            spell_speed (int): Spell speed stat.
            crit_stat (int): Critical hit stat.
            dh_stat (int): Direct hit rate stat.
            weapon_damage (int): Weapon damage
            delay (float): Weapon delay stat, for auto attacks.
            pet_attack_power (int, optional): Pet attack power for AST's Earthly Star, which is attack power - n% power bonus. Defaults to None.
            pet_attack_power_scalar (float, optional): Pet attack power scalar,. Defaults to 1.058.
            pet_attack_power_offset (int, optional): Pet attack power offset, which is also multiplied by pet_attack_power_scalar. Defaults to 0.
            pet_job_attribute (int, optional): Pet job attribute, which is different from the player job attribute. Defaults to 100.
            pet_atk_mod (int, optional): Pet attack modifier, which is the same as the player attack modifier. Defaults to 195.
            level (int, optional): Player level. Defaults to 90.
            intelligence (None, optional): Deprecated argument for intelligence. Defaults to None.
            dexterity (None, optional): Deprecated argument for dexterity. Defaults to None.
            vit (None, optional): Deprecated argument for vitality. Defaults to None.
            tenacity (None, optional): Deprecated argument for tenacity. Defaults to None.
        """
        super().__init__(
            attack_power=mind,
            trait=130,
            main_stat=mind,
            strength=strength,
            det=det,
            crit_stat=crit_stat,
            dh_stat=dh_stat,
            dot_speed_stat=spell_speed,
            auto_speed_stat=400,
            weapon_damage=weapon_damage,
            delay=delay,
            pet_attack_power=pet_attack_power,
            pet_attack_power_scalar=pet_attack_power_scalar,
            pet_attack_power_offset=pet_attack_power_offset,
            pet_job_attribute=pet_job_attribute,
            pet_atk_mod=pet_atk_mod,
            level=level,
        )

        self.auto_trait = 100
        self.atk_mod = 195
        self.job_attribute = 115

        self.dot_speed_stat = spell_speed
        self.auto_speed_stat = 400
        self.add_role("Healer")

        if (
            (dexterity is not None)
            or (intelligence is not None)
            or (vit is not None)
            or (tenacity is not None)
        ):
            warn(
                "Irrelevant main stats (DEX, INT, VIT), and tenacity are no longer required and in a future update will give an error."
            )
        pass


class Tank(BaseStats):
    def __init__(
        self,
        strength: int,
        det: int,
        skill_speed: int,
        tenacity: int,
        crit_stat: int,
        dh_stat: int,
        weapon_damage: int,
        delay: float,
        job: str,
        pet_attack_power: int = None,
        pet_attack_power_scalar: float = 1.,
        pet_attack_power_offset: int = -18,
        pet_job_attribute: int = 100,
        pet_atk_mod: int = 195,
        level: int = 90,
    ) -> None:
        """Set tank stats, most notably the attack modifier.

        Many of the pet stats are used to compute an effective pet attack power, given by:
        effective_pet_attack_power = floor(pet_attack_power_scalar * (pet_attack_power + pet_attack_power_offset))

        Args:
            strength (int): Strength stat.
            det (int): Determination stat.
            skill_speed (int): Skill speed stat.
            tenacity (int): Tenacity stat.
            crit_stat (int): Critical hit stat.
            dh_stat (int): Direct hit rate stat.
            weapon_damage (int): Weapon damage stat.
            delay (float): Weapon delay stat, for auto attacks
            job (str): Job, used to set job modifier. Allowed values are: {"Warrior", "Paladin", "DarkKnight", "Gunbreaker"}
            pet_attack_power (int, optional): Pet attack power, typically attack power - n% party bonus. Defaults to None.
            pet_attack_power_scalar (float, optional): Scalar to multiply pet attack power by. Defaults to 1, the scalar for DRK's Esteem.
            pet_attack_power_offset (int, optional): Pet attack power offset, also multiplied by the pet scalar. Defaults to -18, the offset for DRK's Esteem.
            pet_job_attribute (int, optional): Pet job attribute, which is not the same as the player. Defaults to 100.
            pet_atk_mod (int, optional): Pet attack modifier, which is not the same as the player. Defaults to 195.
            level (int, optional): _description_. Defaults to 90.

        """
        super().__init__(
            attack_power=strength,
            trait=100,
            main_stat=strength,
            det=det,
            tenacity=tenacity,
            crit_stat=crit_stat,
            dh_stat=dh_stat,
            auto_speed_stat=skill_speed,
            dot_speed_stat=skill_speed,
            weapon_damage=weapon_damage,
            delay=delay,
            pet_attack_power=pet_attack_power,
            pet_attack_power_scalar=pet_attack_power_scalar,
            pet_attack_power_offset=pet_attack_power_offset,
            pet_job_attribute=pet_job_attribute,
            pet_atk_mod=pet_atk_mod,
            level=level,
        )

        if (job == "Warrior") | (job == "DarkKnight"):
            self.job_attribute = 105
        elif (job == "Paladin") | (job == "Gunbreaker"):
            self.job_attribute = 100
        else:
            raise ValueError(
                f"Incorrect job of {job} specified. Values of 'Warrior', 'Paladin', 'DarkKnight', or 'Gunbreaker' are allowed "
            )

        self.add_role("Tank")
        if level == 90:
            self.atk_mod = 156
        if level == 80:
            self.atk_mod = 115

        self.dot_speed_stat = skill_speed
        self.auto_speed_stat = skill_speed
        pass


class MagicalRanged(BaseStats):
    def __init__(
        self,
        intelligence: int,
        strength: int,
        det: int,
        spell_speed: int,
        crit_stat: int,
        dh_stat: int,
        weapon_damage: int,
        delay: float,
        pet_attack_power: int = None,
        pet_attack_power_scalar: float = 0.88,
        pet_attack_power_offset: int = -48,
        pet_job_attribute: int = 100,
        pet_atk_mod: int = 195,
        level: int = 90,
    ) -> None:
        """Set stats specific to magical ranged, to compute damage from potency.

        Many of the pet stats are used to compute an effective pet attack power, given by:
        effective_pet_attack_power = floor(pet_attack_power_scalar * (pet_attack_power + pet_attack_power_offset))
        Args:
            intelligence (int): intelligence stat.
            strength (int): strength stat, for auto attacks.
            det (int): determination stat.
            spell_speed (int): spell speed stat.
            crit_stat (int): critical hit stat.
            dh_stat (int): direct hit rate stat
            weapon_damage (int): weapon damage stat.
            delay (float): weapon delay stat, for auto attacks
            pet_attack_power (int, optional): Attack power of pet, typically intelligence without the n% party bonus. Defaults to None.
            pet_attack_power_scalar (float, optional): Scalar to apply to pet attack power to account for pet potency. Defaults to 0.88, the default value for SMN.
            pet_attack_power_offset (int, optional): Attack power amount to add to attack power. Usually -48 to subtract out hidden trait stats. This value is also multiplied by pet_attack_power_scalar. Defaults to -48, the default value for SMN.
            pet_job_attribute (int, optional): Pet job attribute, which is different from that of the player. Defaults to 100.
            pet_atk_mod (int, optional): Pet attack modifier, which is usually the same as the player. Defaults to 195.
            level (int, optional): Player level. Defaults to 90.
        """
        super().__init__(
            attack_power=intelligence,
            trait=130,
            main_stat=intelligence,
            strength=strength,
            det=det,
            crit_stat=crit_stat,
            dh_stat=dh_stat,
            dot_speed_stat=spell_speed,
            auto_speed_stat=400,
            weapon_damage=weapon_damage,
            delay=delay,
            pet_attack_power=pet_attack_power,
            pet_attack_power_scalar=pet_attack_power_scalar,
            pet_attack_power_offset=pet_attack_power_offset,
            pet_job_attribute=pet_job_attribute,
            pet_atk_mod=pet_atk_mod,
            level=level,
        )

        self.add_role("Caster")

        self.auto_trait = 100
        self.dot_speed_stat = spell_speed
        self.auto_speed_stat = 400

        pass


# class PhysicalRanged(BaseStats):
#     def __init__(self, mind, intelligence, vitality, strength, dexterity, det,
#                  skill_speed, spell_speed, tenacity, crit_stat, dh_stat, weapon_damage, delay, level=90) -> None:
#         """
#         Set physical ranged-specific stats with this class like main stat, traits, etc.

#         inputs:
#         mind - int, mind stat
#         intelligence - int, intelligence stat
#         vitality - int, vitality stat
#         strength, - int, strength stat
#         dexterity - int, strength stat
#         det - int, determination stat
#         skill_speed - int, skill speed stat
#         spell_speed - int, spell speed stat
#         tenacity - tenacity stat
#         crit_stat - critical hit stat
#         dh_stat - direct hit rate stat
#         weapon_damage - weapon damage stat
#         delay - weapon delay stat
#         """
#         super().__init__(dexterity, 120, dexterity, mind, intelligence, vitality, strength, dexterity, det,
#                          tenacity, crit_stat, dh_stat, skill_speed, skill_speed, weapon_damage, delay, level=90)

#         self.add_role('Physical Ranged')

#         self.skill_speed = skill_speed
#         self.spell_speed = spell_speed
#         pass


# class Melee(BaseStats):
#     def __init__(self, mind, intelligence, vitality, strength, dexterity,
#                  det, skill_speed, spell_speed, tenacity, crit_stat, dh_stat, weapon_damage, delay, level=90) -> None:
#         """
#         Set melee-specific stats with this class like main stat, traits, etc.

#         inputs:
#         mind - int, mind stat
#         intelligence - int, intelligence stat
#         vitality - int, vitality stat
#         strength, - int, strength stat
#         dexterity - int, strength stat
#         det - int, determination stat
#         skill_speed - int, skill speed stat
#         spell_speed - int, spell speed stat
#         tenacity - tenacity stat
#         crit_stat - critical hit stat
#         dh_stat - direct hit rate stat
#         weapon_damage - weapon damage stat
#         delay - weapon delay stat
#         """
#         super().__init__(strength, 100, strength, mind, intelligence, vitality, strength, dexterity,
#                          det, tenacity, crit_stat, dh_stat, skill_speed, skill_speed, weapon_damage, delay, level=90)

#         self.add_role('Melee')

#         self.skill_speed = skill_speed
#         self.spell_speed = spell_speed

if __name__ == "__main__":
    a = MagicalRanged(
        3369,
        190,
        2136,
        500,
        2399,
        796,
        132,
        3.22,
        3369
    )
    pass
