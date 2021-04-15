import numpy as np
from scipy.stats import truncnorm

class better_truncnorm(object):
    @staticmethod
    def pdf(x, start, end, mean, stdev):
        a, b = (start - mean) / stdev, (end - mean) / stdev
        return truncnorm.pdf(x, a=a, b=b, loc=mean, scale=stdev)

    @staticmethod
    def rvs(start, end, mean, stdev):
        a, b = (start - mean) / stdev, (end - mean) / stdev
        return truncnorm.rvs(a=a, b=b, loc=mean, scale=stdev)

def convert_to_large_num(vec):
    vec[np.isposinf(vec)]=np.finfo(np.float).max
    vec[np.isneginf(vec)]=np.finfo(np.float).min
    return vec