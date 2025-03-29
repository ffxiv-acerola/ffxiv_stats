from pathlib import Path

import numpy as np
import pytest

from ffxiv_stats.jobs import Healer, MagicalRanged, Melee, PhysicalRanged, Tank
from ffxiv_stats.modifiers import pet_defaults
from ffxiv_stats.rate import Rate

test_data_path = Path("tests/data/dawntrail")


class TestDawntrailDamage:
    """Test if base damage values (before random damage rolls and hit types) match between in-game data and values predicted by `ffxiv_stats` for Dawntrail."""

    level = 100

    smn = MagicalRanged(
        intelligence=4135,
        strength=395,
        det=2400,
        spell_speed=677,
        crit_stat=1295,
        dh_stat=1545,
        weapon_damage=137,
        delay=3.12,
        pet_attack_power=4135,
        **pet_defaults["Summoner"][level],
        level=level,
    )

    drk = Tank(
        strength=4248,
        det=2148,
        skill_speed=573,
        tenacity=1338,
        crit_stat=2349,
        dh_stat=636,
        weapon_damage=141,
        delay=2.96,
        pet_attack_power=4248,
        job="DarkKnight",
        **pet_defaults["DarkKnight"][level],
        level=level,
    )

    def test_auto_attack(self):
        """Test that DRK's in-game auto attack base damage is within 2 damage points of the predicted value."""
        drk_auto = np.genfromtxt(test_data_path / "drk_auto_90p.csv")
        assert self.drk.auto_attack_d2(90) == pytest.approx(drk_auto.mean(), abs=2)

    def test_direct_damage(self):
        pass

    def test_magic_dot(self):
        pass

    def test_ast_star(self):
        pass

    def test_drk_pet_420(self):
        """Test if predicted and in-game base damage values for Esteem (420 potency actions) are within 1% of each other."""
        dark_pet_420 = np.genfromtxt(test_data_path / "drk_pet_420p.csv")
        assert self.drk.pet_direct_d2(420) == pytest.approx(
            dark_pet_420.mean(), rel=0.01
        )

    def test_smn_pet_150(self):
        """Test if predicted and in-game base damage values for Wyrmwave and Scarlet Flame are within 1% of each other."""
        smn_150 = np.genfromtxt(test_data_path / "smn_pet_150p.csv")
        assert self.smn.pet_direct_d2(150) == pytest.approx(smn_150.mean(), rel=0.01)

    def test_smn_pet_160(self):
        """Test if predicted and in-game base damage values for Wyrmwave and Scarlet Flame are within 1% of each other."""
        smn_160 = np.genfromtxt(test_data_path / "smn_pet_160p.csv")
        assert self.smn.pet_direct_d2(160) == pytest.approx(smn_160.mean(), rel=0.01)
