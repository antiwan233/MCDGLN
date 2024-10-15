from source.dataset.load_abide import load_abide
from utils import retain_top_percent
import numpy as np


def main():

    tc, corr, labels, sites = load_abide('aal')

    test_corr = corr[16]

    threshold_corr = retain_top_percent(test_corr, percent=0.2)

    print((threshold_corr == test_corr).sum())


if __name__ == '__main__':

    main()
