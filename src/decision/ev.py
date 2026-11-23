import numpy as np

from src.config import settings


def lift(p, biz=None) -> np.ndarray:
    biz = biz or settings.business
    ramp = max(biz.lift_ramp_prob, 1e-9)
    return biz.max_lift * np.minimum(1.0, np.asarray(p, dtype=float) / ramp)


def expected_value(p, biz=None) -> np.ndarray:
    biz = biz or settings.business
    p = np.asarray(p, dtype=float)
    gained = lift(p, biz) * (biz.sale_value - biz.discount_cost)
    wasted = p * biz.discount_cost
    return gained - wasted - biz.show_cost