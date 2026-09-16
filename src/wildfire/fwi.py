"""
The Canadian Forest Fire Weather Index (FWI) System, vectorised.

Van Wagner, C.E. (1987). *Development and structure of the Canadian Forest Fire
Weather Index System.* Forestry Technical Report 35, Canadian Forestry Service.
The equations and constants below follow that report and the reference
implementation in the `cffdrs` R package; `tests/test_fwi.py` holds them to the
daily values the Canadian Wildland Fire Information System publishes for its
weather stations.

Six numbers come out of four noon weather readings:

    FFMC  Fine Fuel Moisture Code  - litter and fine fuels, hours to dry
    DMC   Duff Moisture Code       - loosely compacted organic layers, days
    DC    Drought Code             - deep compact organic layers, weeks to months
    ISI   Initial Spread Index     - FFMC and wind: how fast a fire would spread
    BUI   Buildup Index            - DMC and DC: how much fuel is available
    FWI   Fire Weather Index       - ISI and BUI: fire intensity
    DSR   Daily Severity Rating    - FWI rescaled to be additive over a season

The three moisture codes are *state*: each day's value is yesterday's value moved
by today's weather. That is why the same function serves training (twenty years
of history for every grid cell) and the live forecast (today's official codes
carried forward with forecast weather): nothing about the arithmetic changes,
only where the starting codes come from.

Every function takes numpy arrays (one element per location) so a whole grid
steps forward one day in a single call.
"""

from __future__ import annotations

import numpy as np

# Start-of-season codes when nothing better is known (Van Wagner 1987).
FFMC_START, DMC_START, DC_START = 85.0, 6.0, 15.0

# Effective day length for DMC and DC drying, by month, for latitudes above 30 N.
DMC_DAY_LENGTH = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
DC_DAY_LENGTH = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])


def _arrays(*values):
    return [np.asarray(v, dtype=float) for v in values]


def ffmc(ffmc_yesterday, temp, rh, wind, rain):
    """Fine Fuel Moisture Code from yesterday's FFMC and today's noon weather.

    temp in C, rh in %, wind in km/h at 10 m, rain as the 24 h total in mm."""
    f0, temp, rh, wind, rain = _arrays(ffmc_yesterday, temp, rh, wind, rain)
    rh = np.clip(rh, 0.0, 100.0)
    wind = np.maximum(wind, 0.0)
    rain = np.maximum(rain, 0.0)

    mo = 147.2 * (101.0 - f0) / (59.5 + f0)
    wetting = rain > 0.5
    rf = np.where(wetting, rain - 0.5, 1.0)
    gain = 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
    saturated = mo > 150.0
    wet = np.where(saturated, mo + gain + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), mo + gain)
    mo = np.where(wetting, np.minimum(wet, 250.0), mo)

    ed = 0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) \
        + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    ew = 0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) \
        + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))

    ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - (rh / 100.0) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    drying = ed + (mo - ed) * 10.0 ** (-kd)

    k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) \
        + 0.0694 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
    kw = k1 * 0.581 * np.exp(0.0365 * temp)
    wetting_up = ew - (ew - mo) * 10.0 ** (-kw)

    m = np.where(mo > ed, drying, np.where(mo < ew, wetting_up, mo))
    return np.clip(59.5 * (250.0 - m) / (147.2 + m), 0.0, 101.0)


def dmc(dmc_yesterday, temp, rh, rain, month):
    """Duff Moisture Code. month is 1-12, per location or a scalar."""
    p0, temp, rh, rain = _arrays(dmc_yesterday, temp, rh, rain)
    month = np.asarray(month, dtype=int)
    rh = np.clip(rh, 0.0, 100.0)
    rain = np.maximum(rain, 0.0)
    p0 = np.maximum(p0, 0.0)

    t = np.maximum(temp, -1.1)
    rk = 1.894 * (t + 1.1) * (100.0 - rh) * DMC_DAY_LENGTH[month - 1] * 1e-4

    wetting = rain > 1.5
    rw = 0.92 * rain - 1.27
    wmi = 20.0 + 280.0 / np.exp(0.023 * p0)
    safe_p0 = np.maximum(p0, 1e-9)
    b = np.where(p0 <= 33.0, 100.0 / (0.5 + 0.3 * p0),
                 np.where(p0 <= 65.0, 14.0 - 1.3 * np.log(safe_p0), 6.2 * np.log(safe_p0) - 17.2))
    wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
    wet = 43.43 * (5.6348 - np.log(np.maximum(wmr - 20.0, 1e-9)))
    pr = np.maximum(np.where(wetting, wet, p0), 0.0)
    return np.maximum(pr + rk, 0.0)


def dc(dc_yesterday, temp, rain, month):
    """Drought Code. month is 1-12, per location or a scalar."""
    d0, temp, rain = _arrays(dc_yesterday, temp, rain)
    month = np.asarray(month, dtype=int)
    rain = np.maximum(rain, 0.0)
    d0 = np.maximum(d0, 0.0)

    t = np.maximum(temp, -2.8)
    pe = np.maximum((0.36 * (t + 2.8) + DC_DAY_LENGTH[month - 1]) / 2.0, 0.0)

    wetting = rain > 2.8
    rw = 0.83 * rain - 1.27
    smi = 800.0 * np.exp(-d0 / 400.0)
    wet = np.maximum(d0 - 400.0 * np.log(1.0 + 3.937 * np.maximum(rw, 0.0) / smi), 0.0)
    dr = np.where(wetting, wet, d0)
    return np.maximum(dr + pe, 0.0)


def isi(ffmc_today, wind):
    """Initial Spread Index."""
    f, wind = _arrays(ffmc_today, wind)
    fm = 147.2 * (101.0 - f) / (59.5 + f)
    sf = 19.115 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)
    return sf * np.exp(0.05039 * np.maximum(wind, 0.0))


def bui(dmc_today, dc_today):
    """Buildup Index."""
    p, d = _arrays(dmc_today, dc_today)
    denominator = p + 0.4 * d
    safe = np.where(denominator > 0, denominator, 1.0)
    low = np.where(denominator > 0, 0.8 * d * p / safe, 0.0)
    high = p - (1.0 - 0.8 * d / safe) * (0.92 + (0.0114 * p) ** 1.7)
    return np.maximum(np.where(p <= 0.4 * d, low, high), 0.0)


def fwi(isi_today, bui_today):
    """Fire Weather Index."""
    r, u = _arrays(isi_today, bui_today)
    fd = np.where(u <= 80.0, 0.626 * u ** 0.809 + 2.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * u)))
    bb = 0.1 * r * fd
    safe = np.maximum(bb, 1.0 + 1e-12)
    return np.where(bb <= 1.0, bb, np.exp(2.72 * (0.434 * np.log(safe)) ** 0.647))


def dsr(fwi_today):
    """Daily Severity Rating."""
    return 0.0272 * np.asarray(fwi_today, dtype=float) ** 1.77


def step(state, temp, rh, wind, rain, month):
    """Advance every location one day.

    state is a dict of arrays with keys ffmc, dmc, dc (yesterday's codes). Returns
    today's full set: ffmc, dmc, dc, isi, bui, fwi, dsr."""
    f = ffmc(state["ffmc"], temp, rh, wind, rain)
    p = dmc(state["dmc"], temp, rh, rain, month)
    d = dc(state["dc"], temp, rain, month)
    spread = isi(f, wind)
    buildup = bui(p, d)
    index = fwi(spread, buildup)
    return {"ffmc": f, "dmc": p, "dc": d, "isi": spread, "bui": buildup, "fwi": index, "dsr": dsr(index)}


def start_state(n: int) -> dict:
    """Default start-of-season codes for n locations."""
    return {"ffmc": np.full(n, FFMC_START), "dmc": np.full(n, DMC_START), "dc": np.full(n, DC_START)}
