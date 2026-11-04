"""Motion-blur profile selection for one image or one paired-scene group."""

import math


def sample_profile(rng, probability, exposures_s, speeds_m_s):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("motion blur probability must be between 0 and 1")
    if rng.random() >= probability:
        return {"enabled": False, "exposure_s": None, "speed_m_s": 0.0, "direction_rad": 0.0}
    return {
        "enabled": True,
        "exposure_s": rng.choice(tuple(exposures_s)),
        "speed_m_s": rng.uniform(*speeds_m_s),
        "direction_rad": rng.uniform(0, math.tau),
    }


def quality_bucket(max_endpoint_displacement_px, enabled):
    if not enabled or max_endpoint_displacement_px is None:
        return "clear" if not enabled else "unverified"
    # The motion streak spans both exposure endpoints.
    streak = 2 * max_endpoint_displacement_px
    if streak < 0.5:
        return "clear"
    return "slight_blur" if streak <= 4.0 else "significant_blur"
