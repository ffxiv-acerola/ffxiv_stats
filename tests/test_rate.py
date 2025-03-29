from ffxiv_stats.rate import Rate


class TestDawntrailRates:
    critical_hit = 2000
    direct_hit = 1250
    determination = 2000
    level = 100
    r = Rate(critical_hit, direct_hit, level)

    def test_critical_hit_damage_multiplier(self) -> None:
        assert self.r.l_c == 1513

    def test_critical_hit_rate(self) -> None:
        assert self.r.crit_prob() == 0.163

    def test_direct_hit_rate(self) -> None:
        assert self.r.direct_hit_prob() == 0.164

    def test_hit_type_probability(self) -> None:
        assert (
            (self.r.p[0] == 0.699732)
            & (self.r.p[1] == 0.136268)
            & (self.r.p[2] == 0.137268)
            & (self.r.p[3] == 0.026732)
        )

    def test_auto_crit(self) -> None:
        """Test that a guaranteed critical hit correctly applies a critical hit rate buff as a damage buff.
        """
        pass

    def test_auto_dh(self) -> None:
        """Test that a guaranteed direct hit has above-base direct hit stat count as a damage buff."""
        pass

    def test_auto_cdh_with_crit_buff(self) -> None:
        """Test that a guaranteed critical-direct hit correctly applies a critical hit rate buff as a damage buff.
        """
        pass

    def test_auto_cdh_with_dh_buff(self) -> None:
        """Test that a guaranteed critical-direct hit correctly applies a direct hit rate buff as a damage buff.
        """
        pass

    def test_auto_cdh_with_crit_dh_buff(self) -> None:
        """Test that a guaranteed critical-direct hit correctly applies both a critical hit rate buff and direct hit rate buff as damage buffs.
        """
        pass

    def test_auto_dh_no_dh(self) -> None:
        """Test a guaranteed critical-direct hit, but there is no melded direct hit stat.
        A base direct hit rate stat should grant no damage buff.
        """
        pass