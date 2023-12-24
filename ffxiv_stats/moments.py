import pandas as pd
import numpy as np
from scipy.stats import multinomial, skewnorm
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

class Support():

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

        self.normal_supp = self.get_support('normal')
        self.crit_supp = self.get_support('critical')
        self.dir_supp = self.get_support('direct')
        self.crit_dir_supp = self.get_support('critical-direct')
        pass

    def buff_check(self, buffs):
        if buffs is None or len(buffs) == 0:
            return 1
        else:
            return np.product(buffs)

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
            lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2)+1
            damage_range = np.arange(lower, upper)
            
            if hit_type == 'normal':
                support = np.floor(damage_range * self.buff_prod)

            elif hit_type == 'critical':
                support = np.floor(np.floor(np.floor(damage_range * self.l_c) / 1000) * self.buff_prod)

            elif hit_type == 'direct':
                support = np.floor(np.floor(np.floor(damage_range * self.l_d) / 100) * self.buff_prod)
            
            elif hit_type == 'critical-direct':
                ch_dmg = np.floor(np.floor(damage_range * self.l_c) / 1000)
                support = np.floor(np.floor(np.floor(ch_dmg * self.l_d) / 100) * self.buff_prod)
            
            else:
                print('incorrect input')

        # Attack is not a DoT
        else:
            if hit_type == 'normal':
                lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2)+1

            elif hit_type == 'critical':
                lower, upper = np.floor(0.95 * self.ch_dmg_modifier()), np.floor(1.05 * self.ch_dmg_modifier())+1   

            elif hit_type == 'direct':
                lower, upper = np.floor(0.95 * self.dh_dmg_modifier()), np.floor(1.05 * self.dh_dmg_modifier())+1

            elif hit_type == 'critical-direct':
                lower, upper = np.floor(0.95 * self.cdh_dmg_modifier()), np.floor(1.05 * self.cdh_dmg_modifier())+1

            damage_range = np.arange(lower, upper)
            support = np.floor(damage_range * self.buff_prod)

        return support


class ActionMoments(Support):

    def __init__(self, action_df, t) -> None:
        """
        Compute moments for a action landing n_hits

        inputs: 
        action_df - pandas dataframe with the columns: 
                    n: int, number of hits.
                    p: list of probability lists, in order [p_NH, p_CH, p_DH, p_CDH].
                    d2: int, base damage value of action before any variability.
                    l_c: int, damage multiplier for a critical hit. 
                              Value should be in the thousands (1250 -> 125% crit buff).
                    buffs: list of buffs present. A 10% buff should be represented as [1.10], no buff as [1].
                    is_dot: boolean or 0/1, whether the action is a damage over time effect.
        """

        column_check = set(["n", "p", "d2", "l_c", "buffs", "is_dot"])
        if isinstance(action_df, pd.core.series.Series):
            supplied_columns = action_df.index
        else:
            supplied_columns = action_df.columns
        missing_columns = column_check - set(supplied_columns)
        if len(missing_columns) != 0:
            raise ValueError(f"The following column(s) are missing from `rotation_df`: {*missing_columns,}. Please refer to the docstring and add these field(s) or double check the spelling.")

        self.n = action_df['n']
        self.p = action_df['p']
        self.t = t
        if 'action_name' in action_df:
            self.action_name = action_df['action_name']
        # All possible hit types
        self.x = self.hit_type_combos(self.n)
        # Corresponding multinomial weight
        self.w = multinomial(self.n, self.p).pmf(self.x)

        Support.__init__(self, action_df['d2'], action_df['l_c'], bool(action_df['is_dot']), action_df['buffs'])
        
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
        self._Z_D =  np.sum(self.dir_supp)
        self._Z_D2 = np.sum(self.dir_supp**2)
        self._Z_D3 = np.sum(self.dir_supp**3)

        self._S_CD = self.crit_dir_supp.size
        self._Z_CD =  np.sum(self.crit_dir_supp)
        self._Z_CD2 = np.sum(self.crit_dir_supp**2)
        self._Z_CD3 = np.sum(self.crit_dir_supp**3)

        self._first_moment = self._get_first_moment()
        self._second_moment = self._get_second_moment()
        self._third_moment = self._get_third_moment()

        self.mean = self._first_moment
        self.variance = self.get_action_variance()
        self.skewness = self.get_action_skewness()

        # Convert from total damage to DPS
        self.mean /= self.t
        self.variance /= self.t**2

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
        import itertools, operator
        hit_list = []
        for cuts in itertools.combinations_with_replacement(range(n+1), 3):
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
        first_deriv = ((self.x[:,1]*self._Z_C)/self._S_C + (self.x[:,3]*self._Z_CD)/self._S_CD + (self.x[:,2]*self._Z_D)/self._S_D + (self.x[:,0]*self._Z_N)/self._S_N)
        return np.dot(self.w, first_deriv)

    def _get_second_moment(self):
        """
        Compute the second moment for an action landing n hits
        """
        second_deriv = ((self.x[:,1]**2*self._Z_C**2)/self._S_C**2 + (self.x[:,3]**2*self._Z_CD**2)/self._S_CD**2 \
                        + self.x[:,3]*(-(self._Z_CD**2/self._S_CD**2) + self._Z_CD2/self._S_CD) + (self.x[:,2]**2*self._Z_D**2)/self._S_D**2 \
                        + self.x[:,1]*(-(self._Z_C**2/self._S_C**2) + self._Z_C2/self._S_C + (2*self.x[:,3]*self._Z_C*self._Z_CD)/(self._S_C*self._S_CD) \
                        + (2*self.x[:,2]*self._Z_C*self._Z_D)/(self._S_C*self._S_D)) + self.x[:,2]*((2*self.x[:,3]*self._Z_CD*self._Z_D)/(self._S_CD*self._S_D) \
                        - self._Z_D**2/self._S_D**2 + self._Z_D2/self._S_D) + (self.x[:,0]**2*self._Z_N**2)/self._S_N**2 \
                        + self.x[:,0]*((2*self.x[:,1]*self._Z_C*self._Z_N)/(self._S_C*self._S_N) + (2*self.x[:,3]*self._Z_CD*self._Z_N)/(self._S_CD*self._S_N)\
                        + (2*self.x[:,2]*self._Z_D*self._Z_N)/(self._S_D*self._S_N) - self._Z_N**2/self._S_N**2 + self._Z_N2/self._S_N))
        return np.dot(self.w, second_deriv)

    def _get_third_moment(self):
        """
        Compute the third moment for an action landing n hits.
        """

        third_deriv = ((self.x[:,1]**3*self._Z_C**3)/self._S_C**3 + (self.x[:,3]**3*self._Z_CD**3)/self._S_CD**3 \
                        + self.x[:,3]**2*((-3*self._Z_CD**3)/self._S_CD**3 + (3*self._Z_CD*self._Z_CD2)/self._S_CD**2) \
                        + self.x[:,3]*((2*self._Z_CD**3)/self._S_CD**3 - (3*self._Z_CD*self._Z_CD2)/self._S_CD**2 + self._Z_CD3/self._S_CD) \
                        + (self.x[:,2]**3*self._Z_D**3)/self._S_D**3 + self.x[:,1]**2*((-3*self._Z_C**3)/self._S_C**3 \
                        + (3*self._Z_C*self._Z_C2)/self._S_C**2 + (3*self.x[:,3]*self._Z_C**2*self._Z_CD)/(self._S_C**2*self._S_CD) 
                        + (3*self.x[:,2]*self._Z_C**2*self._Z_D)/(self._S_C**2*self._S_D)) + self.x[:,2]**2*((3*self.x[:,3]*self._Z_CD*self._Z_D**2)/(self._S_CD*self._S_D**2) \
                        - (3*self._Z_D**3)/self._S_D**3 + (3*self._Z_D*self._Z_D2)/self._S_D**2) + self.x[:,1]*((2*self._Z_C**3)/self._S_C**3 \
                        - (3*self._Z_C*self._Z_C2)/self._S_C**2 + self._Z_C3/self._S_C + (3*self.x[:,3]**2*self._Z_C*self._Z_CD**2)/(self._S_C*self._S_CD**2) \
                        + self.x[:,3]*((-3*self._Z_C**2*self._Z_CD)/(self._S_C**2*self._S_CD) + (3*self._Z_C2*self._Z_CD)/(self._S_C*self._S_CD) \
                        - (3*self._Z_C*self._Z_CD**2)/(self._S_C*self._S_CD**2) + (3*self._Z_C*self._Z_CD2)/(self._S_C*self._S_CD)) + (3*self.x[:,2]**2*self._Z_C*self._Z_D**2)/(self._S_C*self._S_D**2) \
                        + self.x[:,2]*((-3*self._Z_C**2*self._Z_D)/(self._S_C**2*self._S_D) + (3*self._Z_C2*self._Z_D)/(self._S_C*self._S_D) + (6*self.x[:,3]*self._Z_C*self._Z_CD*self._Z_D)/(self._S_C*self._S_CD*self._S_D)\
                        - (3*self._Z_C*self._Z_D**2)/(self._S_C*self._S_D**2) + (3*self._Z_C*self._Z_D2)/(self._S_C*self._S_D))) + self.x[:,2]*((3*self.x[:,3]**2*self._Z_CD**2*self._Z_D)/(self._S_CD**2*self._S_D) \
                        + (2*self._Z_D**3)/self._S_D**3 - (3*self._Z_D*self._Z_D2)/self._S_D**2 + self.x[:,3]*((-3*self._Z_CD**2*self._Z_D)/(self._S_CD**2*self._S_D) \
                        + (3*self._Z_CD2*self._Z_D)/(self._S_CD*self._S_D) - (3*self._Z_CD*self._Z_D**2)/(self._S_CD*self._S_D**2) + (3*self._Z_CD*self._Z_D2)/(self._S_CD*self._S_D)) + self._Z_D3/self._S_D) \
                        + (self.x[:,0]**3*self._Z_N**3)/self._S_N**3 + self.x[:,0]**2*((3*self.x[:,1]*self._Z_C*self._Z_N**2)/(self._S_C*self._S_N**2) \
                        + (3*self.x[:,3]*self._Z_CD*self._Z_N**2)/(self._S_CD*self._S_N**2) + (3*self.x[:,2]*self._Z_D*self._Z_N**2)/(self._S_D*self._S_N**2) \
                        - (3*self._Z_N**3)/self._S_N**3 + (3*self._Z_N*self._Z_N2)/self._S_N**2) + self.x[:,0]*((3*self.x[:,1]**2*self._Z_C**2*self._Z_N)/(self._S_C**2*self._S_N) \
                        + (3*self.x[:,3]**2*self._Z_CD**2*self._Z_N)/(self._S_CD**2*self._S_N) + (3*self.x[:,2]**2*self._Z_D**2*self._Z_N)/(self._S_D**2*self._S_N) \
                        + (2*self._Z_N**3)/self._S_N**3 - (3*self._Z_N*self._Z_N2)/self._S_N**2 + self.x[:,1]*((-3*self._Z_C**2*self._Z_N)/(self._S_C**2*self._S_N) \
                        + (3*self._Z_C2*self._Z_N)/(self._S_C*self._S_N) + (6*self.x[:,3]*self._Z_C*self._Z_CD*self._Z_N)/(self._S_C*self._S_CD*self._S_N) + (6*self.x[:,2]*self._Z_C*self._Z_D*self._Z_N)/(self._S_C*self._S_D*self._S_N) \
                        - (3*self._Z_C*self._Z_N**2)/(self._S_C*self._S_N**2) + (3*self._Z_C*self._Z_N2)/(self._S_C*self._S_N)) + self.x[:,3]*((-3*self._Z_CD**2*self._Z_N)/(self._S_CD**2*self._S_N) \
                        + (3*self._Z_CD2*self._Z_N)/(self._S_CD*self._S_N) - (3*self._Z_CD*self._Z_N**2)/(self._S_CD*self._S_N**2) + (3*self._Z_CD*self._Z_N2)/(self._S_CD*self._S_N)) \
                        + self.x[:,2]*((6*self.x[:,3]*self._Z_CD*self._Z_D*self._Z_N)/(self._S_CD*self._S_D*self._S_N) - (3*self._Z_D**2*self._Z_N)/(self._S_D**2*self._S_N) + (3*self._Z_D2*self._Z_N)/(self._S_D*self._S_N) \
                        - (3*self._Z_D*self._Z_N**2)/(self._S_D*self._S_N**2) + (3*self._Z_D*self._Z_N2)/(self._S_D*self._S_N)) + self._Z_N3/self._S_N))

        return np.dot(self.w, third_deriv)
    
    def get_action_variance(self):
        return self._second_moment - self.mean**2

    def get_action_skewness(self):
        return (self._third_moment - 3*self.mean*self.variance - self.mean**3) / self.variance**(3./2.)

class Rotation():

    def __init__(self, rotation_df, t, convolve_all=False, delta:int=250) -> None:
        """
        Get damage variability for a rotation.

        Inputs:
        rotation_df: rotation dataframe with the following columns and types:
                     action_name: str, unique name of an action. Unique action depends on `buffs`, `p`, and `l_c` present.
                     base_action: str, name of an action ignoring buffs. For example, Glare III with chain stratagem
                                       and Glare III with mug will have different `action_names`, but the same base_action.
                                       Used for grouping actions together.
                     n: int, number of hits.
                     p: list of probability lists, in order [p_NH, p_CH, p_DH, p_CDH].
                     d2: int, base damage value of action before any variability.
                     l_c: int, damage multiplier for a critical hit. 
                               Value should be in the thousands (1250 -> 125% crit buff).
                     buffs: list of buffs present. A 10% buff should is represented as [1.10]. No buffs can be represented at [1] or None.
                     is_dot: boolean or 0/1, whether the action is a damage over time effect.
        t: float, time elapsed in seconds. Set t=1 to get damage dealt instead of DPS.
        convolve_all: bool, whether to compute all DPS distributions by convolutions (normally actions with large n can be computed with a skew normal distribution).
        delta: int, step size for damage grid used in convolving unique action distributions together.
        """
        column_check = set(["base_action", "action_name"])
        missing_columns = column_check - set(rotation_df.columns)
        if len(missing_columns) != 0:
            raise ValueError(f"The following column(s) are missing from `rotation_df`: {*missing_columns,}. Please refer to the docstring and add these field(s) or double check the spelling.")
                
        self.rotation_df = rotation_df
        self.t = t
        # Deprecated/currently unused
        self.convolve_all = convolve_all
        # Damage is discretized by this much.
        # Bigger number = faster but larger discretization error
        # Smaller number = slower but more accurate.
        self.delta = delta

        self.action_moments = [ActionMoments(row, t) for _, row in rotation_df.iterrows()]
        self.action_names = rotation_df['action_name'].tolist()
        self.action_means = np.array([x.mean for x in self.action_moments]) 
        self.action_variances = np.array([x.variance for x in self.action_moments])
        self.action_std = np.sqrt(self.action_variances)
        self.action_skewness = np.array([x.skewness for x in self.action_moments]) 

        self.rotation_mean = np.sum(self.action_means)
        self.rotation_variance = np.sum(self.action_variances)
        self.rotation_std = np.sqrt(self.rotation_variance)
        # Need just the numerator of Pearson's skewness, which is why we multiply by the action variances inside the sum
        self.rotation_skewness = np.sum(self.action_skewness * self.action_variances**(3/2)) / np.sum(self.action_variances)**(3/2) 

        self.compute_dps_distributions()

        pass

    def compute_dps_distributions(self) -> None:
        """
        Compute and set the support and PMF of DPS distributions.

        This method is broken into 3 sections
        (i) Individual actions (remember Action A with Buff 1 is distinct from Action A with Buff 2).
        (ii) Unique actions (Action A with Buff 1 and Action A with Buff 2 are group together now).
        (iii) The entire rotation.
        """
        # Specifics on convolving everything together because there are quite a few nuances to
        # do things efficiently while still being correct.
        # All damage distributions is convolved together using damage and not DPS. 
        # The supports of each distribution must be on the same grid for the convolution to correctly
        # correspond to a sum of random variables. Converting to DPS usually ends up with floats, so
        # dealing with integer values of damage is a much more convenient unit to work in.

        # At first, we just keep everything in terms of damage, and the supports are just 
        # all integers from the lower to upper bound. However, this makes the convolutions very expensive.
        # The computational cost is N log N, where N is the number of integers between the lower (all hits normal)
        # and upper (all hits critical-direct) bound. This can get very large (N ~ 1e7-1e8) and become  
        # computationally expensive, even with N log N complexity. Instead of working in steps of 1 damage, 
        # we can work in higher steps of damage, like 100/1000/10000/etc, by interpolating the damage 
        # distributions to a coarser grid. This process is referred to as "coarsening".

        # The major consideration for coarsening is when to coarsen and by how much. 
        # Coarsening leads to a greater reduction in computational efficiency when N becomes large.
        # All action distributions are initially convolved in steps of 1 damage n_hit times.
        # Unique action distributions are also convolved in steps of 1 damage, and then coarsened.

        # The action with the smallest damage span will limit how much the support can be coarsened by.
        # The auto-attacks of a WHM only span 10s of damage, but their Afflatus Misery action can span 10,000s. 
        # This is a somewhat unique case, which also makes the argument that auto-attacks can be ignored.
        # By default, damage is discretized in steps of 250, which seemed to still give good accuracy.
        # This also means that actions with very low damage spans are ignored, like healer auto attacks.
        # This wont have a large impact on the rotation damage distribution.
        # A future update might work on dynamically setting this value, or allow for different spacings,
        # which are unified at the very end. 

        # section (i), individual actions
        self.action_dps_support = [None] * self.action_means.size
        self.action_dps_distributions = [None] * self.action_means.size
        self.rotation_dps_distribution = None
        for a in range(self.action_means.size):
            self.action_dps_support[a], self.action_dps_distributions[a] = self.convolve_pmf(a)

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
                action_low_high[idx, :] = np.array([self.action_dps_support[action_idx].min(), self.action_dps_support[action_idx].max()])
            

            if len(action_idx_list) == 1:
                action_dps_distribution = self.action_dps_distributions[action_idx_list[0]]

            elif len(action_idx_list) > 1:
                action_dps_distribution = fftconvolve(self.action_dps_distributions[action_idx_list[0]], 
                                                    self.action_dps_distributions[action_idx_list[1]])

            if len(action_idx_list) > 2:
                for idx in range(1, len(action_idx_list)-1):
                    action_dps_distribution = fftconvolve(action_dps_distribution, 
                                                          self.action_dps_distributions[action_idx_list[idx+1]])           
            
            # Coarsen support in prep for rotation distribution
            uncoarsened_supported = np.arange(action_low_high[:,0].sum(), action_low_high[:,1].sum() + 1, step=1)
            coarsened_support = np.arange(action_low_high[:,0].sum(), action_low_high[:,1].sum() + self.delta, step=self.delta)
            action_dps_distribution = np.interp(coarsened_support, uncoarsened_supported, action_dps_distribution)

            self.unique_actions_distribution[name] = {'support': coarsened_support, 'dps_distribution': action_dps_distribution}


        # Section (iii) whole rotation
        rotation_lower_bound = np.array([v['support'][0] for _, v in self.unique_actions_distribution.items()]).sum()
        rotation_upper_bound = np.array([v['support'][-1] for _, v in self.unique_actions_distribution.items()]).sum()
        
        # `self.rotation_dps_distribution` needs to first be defined by convolving the first two unique actions together
        # then we can loop starting at the second index.
        if len(self.action_moments) > 1:
            self.rotation_dps_distribution = fftconvolve(self.unique_actions_distribution[unique_action_names[0]]['dps_distribution'],
                                                         self.unique_actions_distribution[unique_action_names[1]]['dps_distribution'])
        # Special case if theres only one action, just return the first element.
        else:
            self.rotation_dps_distribution = self.unique_actions_distribution[unique_action_names[0]]['dps_distribution']

        # Now loop
        if len(self.action_moments) > 2:
            for a in range(2, len(unique_action_names)):
                self.rotation_dps_distribution = fftconvolve(self.unique_actions_distribution[unique_action_names[a]]['dps_distribution'], 
                                                             self.rotation_dps_distribution)

        # Create support and convert to DPS
        self.rotation_dps_support = np.arange(rotation_lower_bound, rotation_upper_bound+self.delta, step=self.delta).astype(float) / self.t
        # And renormalize the DPS distribution
        self.rotation_dps_distribution /= np.trapz(self.rotation_dps_distribution, self.rotation_dps_support)

        # Now all the damage distributions have been computed, can convert to DPS
        # action dps distributions
        for idx in range(len(self.action_dps_distributions)):
            self.action_dps_support[idx] /= self.t
            self.action_dps_distributions[idx] /= np.trapz(self.action_dps_distributions[idx], self.action_dps_support[idx])

        for u in unique_action_names:
            self.unique_actions_distribution[u]['support'] /= self.t
            self.unique_actions_distribution[u]['dps_distribution'] /= np.trapz(self.unique_actions_distribution[u]['dps_distribution'], 
                                                                                self.unique_actions_distribution[u]['support'])

        pass

    @classmethod
    def moments_to_skew_norm(self, mean, variance, skewness):
        """
        Converts the mean, variance, and Pearson's skewness to parameters defined by skew normal.
        The parameters are not the same, but can be interconverted: https://en.wikipedia.org/wiki/Skew_normal_distribution
        """

        delta = np.sqrt(np.pi/2 * (np.abs(skewness))**(2/3) / (np.abs(skewness)**(2/3)+((4-np.pi)/2)**(2/3)))
        alpha = np.sign(skewness) * delta / np.sqrt(1 - delta**2)
        omega  = np.sqrt(variance / (1 - 2*delta**2 / np.pi))
        squigma = mean - omega * delta * np.sqrt(2/np.pi)

        return alpha, omega, squigma

    def convolve_pmf(self, action_idx):
        """
        Convolve the single-hit PMF of a action n_hit times to get the exact PMF of an action landing n_hits.
        
        Inputs:
        action_idx - index of an action to compute the damage distribution for

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
                    a = (n+1) // 2
                    b = (n-1) // 2
                
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
            convolve_dict = {
                1: one_hit_pmf
            }

            # How to sum up the partitions to yield n sounds complicated, but there's only 3 cases
            for a in range(len(partition_set) - 1):
                # Self-add: e.g., 1 + 1 = 2
                if partition_set[a] + partition_set[a] == partition_set[a+1]:
                    # print(f"a = {partition_set[a]}: self add {partition_set[a]} + {partition_set[a]}")
                    convolve_dict[partition_set[a+1]] = fftconvolve(convolve_dict[partition_set[a]], convolve_dict[partition_set[a]]) 

                # Add to previous partition: e.g., 7 + 6 = 13
                elif (a > 0) & (partition_set[a-1] + partition_set[a] == partition_set[a+1]):
                    # print(f"a = {partition_set[a]}: prev add {partition_set[a-1]} + {partition_set[a]}")
                    convolve_dict[partition_set[a+1]] = fftconvolve(convolve_dict[partition_set[a-1]], convolve_dict[partition_set[a]])
                
                # Add one: e.g., 6 + 1 = 7
                elif (a > 0 ) & (partition_set[a] + partition_set[0] == partition_set[a+1]):
                    # print(f"a = {partition_set[a]}: 1 add {partition_set[0]} + {partition_set[a]}")
                    convolve_dict[partition_set[a+1]] = fftconvolve(convolve_dict[partition_set[0]], convolve_dict[partition_set[a]])

            return convolve_dict[n]

        # make a shorter variable name cause this long
        action_moment = self.action_moments[action_idx]

        # Define the bounds of the mixture distribution (lowest roll NH and highest roll CDH)
        # Everything is integers, so the bounds can be defined with an arange
        min_roll = np.floor(action_moment.normal_supp[0]).astype(int)
        max_roll = np.floor(action_moment.crit_dir_supp[-1]).astype(int)

        self.one_hit_pmf = np.zeros(max_roll - min_roll + 1)

        # Need to find out how many indices away the start of each hit-type subdistribution is from
        # the lower bound of the mixture distribution.
        ch_offset = int(action_moment.crit_supp[0] - action_moment.normal_supp[0])
        dh_offset = int(action_moment.dir_supp[0] - action_moment.normal_supp[0])
        cdh_offset = int(action_moment.crit_dir_supp[0] - action_moment.normal_supp[0])

        # Set up slices to include gaps
        normal_slice = (action_moment.normal_supp - action_moment.normal_supp[0]).astype(int)
        ch_slice = (action_moment.crit_supp - action_moment.crit_supp[0] + ch_offset).astype(int)
        dh_slice = (action_moment.dir_supp - action_moment.dir_supp[0] + dh_offset).astype(int)
        cdh_slice = (action_moment.crit_dir_supp - action_moment.crit_dir_supp[0] + cdh_offset).astype(int)

        # Mixture distribution defined with multinomial weights
        self.one_hit_pmf[normal_slice] = action_moment.p[0] / action_moment.normal_supp.size
        self.one_hit_pmf[ch_slice] =  action_moment.p[1] / action_moment.crit_supp.size
        self.one_hit_pmf[dh_slice] = action_moment.p[2] / action_moment.dir_supp.size
        self.one_hit_pmf[cdh_slice] = action_moment.p[3] / action_moment.crit_dir_supp.size

        conv_pmf = convolve_by_partitions(self.one_hit_pmf, action_moment.n)
        lowest_roll = int(np.floor(action_moment.normal_supp[0])*action_moment.n)

        dmg_supp = np.arange(lowest_roll, conv_pmf.size + lowest_roll, step=1).astype(float)

        return dmg_supp, conv_pmf

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
            fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=150)
            return_ax = False

        else:
            return_ax = True
        
        for a in range(self.action_means.size):
            ax.plot(self.action_dps_support[a], self.action_dps_distributions[a], label=self.action_names[a])
        ax.set_xlabel('Damage per Second (DPS)')

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
            fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        for _, (name, distributions) in enumerate(self.unique_actions_distribution.items()):
            ax.plot(distributions['support'], distributions['dps_distribution'], label=name, **kwargs)

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
            fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        alpha, omega, squigma = self.moments_to_skew_norm(self.rotation_mean, self.rotation_variance, self.rotation_skewness)
        x = np.linspace(self.rotation_mean - 5 * self.rotation_std, self.rotation_mean + 5 * self.rotation_std, 100)
        y = skewnorm.pdf(x, alpha, squigma, omega)

        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Damage per Second (DPS)')

        if return_ax:
            return ax
        else:
            plt.show()
            pass

if __name__ == "__main__":
    pass