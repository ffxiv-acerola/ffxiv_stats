from ffxiv_stats.rate import Rate
import pytest

@pytest.fixture()
def rate():
    return Rate(2000, 1250, 100)

@pytest.fixture()
def base_rate():
    return Rate(420, 420, 100)

def test_critical_hit_damage_multiplier(rate):
    assert rate.l_c == 1513

def test_critical_hit_rate(rate) -> None:
    assert rate.crit_prob() == 0.163

def test_direct_hit_rate(rate) -> None:
    assert rate.direct_hit_prob() == 0.164

def test_hit_type_probability(rate) -> None:
    assert (
        (rate.p[0] == 0.699732)
        & (rate.p[1] == 0.136268)
        & (rate.p[2] == 0.137268)
        & (rate.p[3] == 0.026732)
    )

@pytest.mark.parametrize(
    "ch_buff, expected",
    [(0, 1.0), (0.1, 1.051) ],
)
def test_guaranteed_critical_damage_multiplier(rate, ch_buff, expected):
    multiplier = rate._guaranteed_critical_damage_multiplier(ch_buff)
    assert multiplier == expected

@pytest.mark.parametrize(
    "dh_buff, det, expected",
    [(0, 420, 1.041), (0, 2000, 1.038), (0.2, 2000, 1.09)],
)
def test_guaranteed_direct_damage_multiplier(rate, dh_buff, det, expected):
    multiplier = rate._guaranteed_direct_damage_multiplier(dh_buff, det)
    assert multiplier == expected

@pytest.mark.parametrize(
    "det",
    [420, 1000],
)
def test_guaranteed_direct_damage_limit(base_rate, det):
    """Test that direct hit rate buff goes to 1.0 in the limit of no direct hit rate stat.
    """
    multiplier = base_rate._guaranteed_direct_damage_multiplier(0, det)
    assert multiplier == 1.0
