import numpy as np
import scipy.stats as stats

dice_scores = [
    0.285593, 0.443195, 0.559724, 0.554802, 0.515171, 0.600692, 0.137934,
    0.657802, 0.277332, 0.622488, 0.367163, 0.366477, 0.492169, 0.095244,
    0.207904, 0.226291, 0.175592, 0.481174, 0.056135, 0.321257, 0.330452,
    0.026811, 0.345153, 0.026093, 0.452058, 0.474938, 0.012432, 0.187141,
    0.525496, 0.407387, 0.115233, 0.662897, 0.421847, 0.087218, 0.390855,
    0.463101, 0.403071, 0.644123, 0.238430, 0.171186
]
nsd_scores = [
    0.385024, 0.572471, 0.864025, 0.746371, 0.708168, 0.698928, 0.185874,
    0.895217, 0.462801, 0.793700, 0.551319, 0.559454, 0.602700, 0.146475,
    0.259798, 0.322252, 0.258692, 0.745429, 0.095618, 0.450897, 0.439479,
    0.060480, 0.464162, 0.046980, 0.578540, 0.659422, 0.031202, 0.237080,
    0.626822, 0.585738, 0.209465, 0.837751, 0.710804, 0.282902, 0.486207,
    0.556580, 0.510264, 0.824306, 0.345648, 0.318599
]

dice_mean = np.mean(dice_scores)
dice_ci = stats.t.interval(0.95, len(dice_scores)-1, loc=dice_mean, scale=stats.sem(dice_scores))
nsd_mean = np.mean(nsd_scores)
nsd_ci = stats.t.interval(0.95, len(nsd_scores)-1, loc=nsd_mean, scale=stats.sem(nsd_scores))

print("Mitjana Dice:", dice_mean)
print("Interval de confiança Dice:", dice_ci)
print("Mitjana NSD:", nsd_mean)
print("Interval de confiança NSD:", nsd_ci)
