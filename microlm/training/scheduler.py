import math


def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        return alpha_max * t / Tw
    if t <= Tc:
        theta = math.pi * (t - Tw) / (Tc - Tw)
        return alpha_min + 0.5 * (1 + math.cos(theta)) * (alpha_max - alpha_min)
    return alpha_min
