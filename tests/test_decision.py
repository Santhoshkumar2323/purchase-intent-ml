import numpy as np

from src.config import BusinessCfg
from src.decision.bands import action_for
from src.decision.ev import expected_value, lift

BIZ = BusinessCfg(discount_cost=300, sale_value=1200, max_lift=0.15,
                  lift_ramp_prob=0.10, show_cost=30)
BANDS = [{"max_prob": 0.029, "action": "no_action"},
         {"max_prob": 0.349, "action": "show_offer"},
         {"max_prob": 1.0, "action": "no_action"}]


def test_lift_is_zero_at_zero_and_capped():
    assert lift(0.0, BIZ) == 0
    assert np.isclose(lift(0.5, BIZ), BIZ.max_lift)


def test_offer_loses_money_at_both_extremes():
    assert expected_value(0.0, BIZ) < 0  
    assert expected_value(0.9, BIZ) < 0   


def test_offer_pays_off_in_the_middle():
    assert expected_value(0.2, BIZ) > 0


def test_action_for_each_band():
    assert action_for(0.01, BANDS) == "no_action"
    assert action_for(0.20, BANDS) == "show_offer"
    assert action_for(0.80, BANDS) == "no_action"
    assert action_for(1.0, BANDS) == "no_action"