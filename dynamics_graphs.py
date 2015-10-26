from math import log10
from scipy.stats import skew
from simulate import TurnoverModel


def main():


    print 'skew:',skew([log10(i) for i in turnover_intervals])


if __name__ == '__main__':
    main()