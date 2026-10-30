"""Retained HIST-STATS definitions; independent historical contract, not a new computation."""
import math, statistics

def quantile(values, p):
    """Linear interpolation at (n-1)*p: Hyndman-Fan type 7."""
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)

def describe(values):
    q1, q3 = quantile(values, .25), quantile(values, .75)
    return dict(median=statistics.median(values), q1=q1, q3=q3,
                iqr=q3-q1, minimum=min(values), maximum=max(values))
