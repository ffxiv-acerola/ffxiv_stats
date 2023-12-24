import numpy as np
from numpy import floor as nf
import pandas as pd

from .moments import Rotation
from .modifiers import level_mod

class BaseStats(Rotation):

    def __init__(self, attack_power, trait, main_stat, mind, intelligence, vitality, strength, dexterity,
                 det, tenacity, crit_stat, dh_stat, dot_speed_stat, auto_speed_stat, weapon_damage, delay, level=90) -> None:
        """
        Base class for converting potency to base damage dealt. Not meant to be used alone.
        Instead use a `ROLE` class, which inherits this class.
        """
        # Level dependent parameters
        # currently for lvl 90
        self.lvl_main = level_mod[level]["lvl_main"]
        self.lvl_sub = level_mod[level]["lvl_sub"]
        self.lvl_div = level_mod[level]["lvl_div"]
        self.job_attribute = 115
        # Set this to 156 if tank
        self.atk_mod = 190
        
        self.main_stat = main_stat
        self.mind = mind
        self.intelligence = intelligence
        self.vitality = vitality
        self.strength = strength
        self.dexterity = dexterity

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

        self.attack_multiplier = self.f_atk()
        self.determination_multiplier = self.f_det()
        self.tenacity_multiplier = self.f_ten()
        self.weapon_damage_multiplier = self.f_wd()
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
        return np.floor((self.lvl_main * self.job_attribute / 1000) + self.weapon_damage)

    def f_atk(self, ap_adjust=0):
        """
        Calculate attack multiplier.

        Inputs
        ap_adjust - int, additional amount to add to attack power (main stat). Used to account for medication.
        """
        return np.floor(self.atk_mod * ((self.attack_power + ap_adjust) - self.lvl_main) / self.lvl_main) + 100

    def f_det(self):
        """
        Calculate determination multiplier.
        """
        return np.floor(140 * (self.det - self.lvl_main) / self.lvl_div + 1000)

    def f_ten(self):
        """
        Calculate tenacity damage multiplier.
        """
        return np.floor(100*(self.tenacity - self.lvl_sub) / self.lvl_div + 1000)
    
    def f_speed_dot(self):
        """
        Calculate speed multiplier for damage over time attacks.
        """
        return np.floor(130 * (self.dot_speed_stat - self.lvl_sub) / self.lvl_div + 1000)

    def f_speed_auto(self):
        """
        Calculate speed multiplier for auto attacks.
        """
        return np.floor(130 * (self.auto_speed_stat - self.lvl_sub) / self.lvl_div + 1000)
    
    def f_auto(self):
        return np.floor(self.f_wd() * self.delay / 3)

    def get_gcd(self):
        # TODO: add GCD (probably not essential?)
        pass

    def attach_rotation(self, rotation_df, t, convolve_all=False, delta=250):
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
                      p: list of probability lists, in order [p_NH, p_CH, p_DH, p_CDH]
                      l_c: int, damage multiplier for a critical hit. 
                                Value should be in the thousands (1250 -> 125% crit buff).
                      buffs: list of buffs present. A 10% buff should is represented as [1.10]. No buffs can be represented at [1] or None.
                      damage_type: str saying the type of damage, {'direct', 'magic-dot', 'physical-dot', 'auto'}
                      main_stat_add: int, how much to add to the main stat (used to account for medication, if present) when computing d2
        """
        column_check = set(["potency", "damage_type"])
        missing_columns = column_check - set(rotation_df.columns)
        if len(missing_columns) != 0:
            raise ValueError(f"The following column(s) are missing from `rotation_df`: {*missing_columns,}. Please refer to the docstring and add these field(s) or double check the spelling.")

        d2 = []
        is_dot = []
        for _, row in rotation_df.iterrows():
            if row['damage_type'] == 'direct':
                d2.append(self.direct_d2(row['potency'], ap_adjust=row['main_stat_add']))
                is_dot.append(0)

            elif row['damage_type'] == 'magic-dot':
                d2.append(self.dot_d2(row['potency'], magic=True, ap_adjust=row['main_stat_add']))
                is_dot.append(1)

            elif row['damage_type'] == 'physical-dot':
                d2.append(self.dot_d2(row['potency'], magic=False, ap_adjust=row['main_stat_add']))
                is_dot.append(1)

            elif row['damage_type'] == 'auto':
                d2.append(self.auto_attack_d2(row['potency']))
                is_dot.append(0)

            else:
                raise ValueError(f"Invalid damage type value of '{row['damage_type']}'. Allow values are ('direct', 'magic-dot', 'physical-dot', 'auto')")

        rotation_df['d2'] = d2
        rotation_df['is_dot'] = is_dot

        super().__init__(rotation_df, t, convolve_all, delta)
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
        if isinstance(self, Healer):
            atk = np.floor(self.atk_mod * ((self.strength + ap_adjust) - self.lvl_main) / self.lvl_main) + 100
        else:
            atk = self.f_atk(ap_adjust)    

        auto_d1 = nf(nf(nf(potency * atk * self.f_det()) / 100) / 1000)
        auto_d2 = nf(nf(nf(nf(nf(nf(nf(nf(auto_d1 * self.f_ten()) / 1000) * self.f_speed_auto()) / 1000) * self.f_auto()) / 100) * self.trait) / 100)
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

        return nf(nf(nf(nf(nf(nf(d1 * self.f_ten()) / 1000) * self.f_wd()) / 100) * self.trait) / 100)
    
    def dot_d2(self, potency, magic=True, ap_adjust=0):
        if magic:
            dot_d1 = nf(nf(nf(nf(nf(nf(potency * self.f_wd()) / 100) * self.f_atk(ap_adjust)) / 100) * self.f_speed_dot()) / 1000)
            return nf(nf(nf(nf(nf(nf(dot_d1 * self.f_det()) / 1000) * self.f_ten()) / 1000) * self.trait) / 100) + 1
        else:
            dot_d1 = nf(nf(nf(potency * self.f_atk() * self.f_det()) / 100) / 1000)
            return nf(nf(nf(nf(nf(nf(nf(nf(dot_d1 * self.f_ten()) / 1000) * self.f_speed_dot()) / 1000) * self.f_wd()) / 100) * self.trait) / 100) + 1

class Healer(BaseStats):
    def __init__(self, mind, intelligence, vitality, strength, dexterity, det, 
                 skill_speed, spell_speed, tenacity, crit_stat, dh_stat, weapon_damage, delay, level=90) -> None:
        """
        Set healer-specific stats with this class like main stat, traits, etc.

        inputs:
        mind - int, mind stat
        intelligence - int, intelligence stat
        vitality - int, vitality stat
        strength, - int, strength stat
        dexterity - int, strength stat
        det - int, determination stat
        skill_speed - int, skill speed stat
        spell_speed - int, spell speed stat
        tenacity - tenacity stat
        crit_stat - critical hit stat
        dh_stat - direct hit rate stat
        weapon_damage - weapon damage stat
        delay - weapon delay stat
        """
        super().__init__(mind, 130, mind, mind, intelligence, vitality, strength, dexterity, det,
                         tenacity, crit_stat, dh_stat, spell_speed, skill_speed, weapon_damage, delay, level=90)

        self.auto_trait = 100
        self.atk_mod = 195
        self.job_attribute = 115

        self.dot_speed_stat = spell_speed
        self.auto_speed_stat = skill_speed
        self.add_role('Healer')
        pass

# class Tank(BaseStats):
#     def __init__(self, mind:int, intelligence:int, vitality:int, strength:int, dexterity:int,
#                  det:int, skill_speed:int, spell_speed:int, tenacity:int, crit_stat:int, dh_stat:int, weapon_damage:int, delay, job:str, level=90) -> None:
#         """
#         Set tank-specific stats with this class like main stat, traits, etc.
#         Most importantly this adjusts the attack modifier.

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

#         if (job.upper() == "WAR") | (job.upper() == "DRK"):
#             self.job_attribute = 105
#         elif (job.upper() == "PLD") | (job.upper() == "GNB"):
#             self.job_attribute = 100
#         else:
#             raise ValueError(f"Incorrect job of {job} specified. Values of 'WAR', 'PLD', 'DRK', or 'GNB' are allowed ")

#         self.add_role('Tank')
#         self.atk_mod = 156
        
#         self.skill_speed = skill_speed
#         self.spell_speed = spell_speed
#         pass


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

# class MagicalRanged(BaseStats):
#     def __init__(self, mind, intelligence, vitality, strength, dexterity, det, 
#                  skill_speed, spell_speed, tenacity, crit_stat, dh_stat, weapon_damage, delay, level=90) -> None:
#         """
#         Set magical ranged-specific stats with this class like main stat, traits, etc.

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
#         super().__init__(intelligence, 130, intelligence, mind, intelligence, vitality, strength, dexterity, det, 
#                          tenacity, crit_stat, dh_stat, spell_speed, skill_speed, weapon_damage, delay, level=90)

#         self.add_role('Caster')
        
#         self.skill_speed = skill_speed
#         self.spell_speed = spell_speed



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
    pass