from math import log10
import datetime

from scipy.stats import skew
import numpy as np
from simulate import TurnoverModel


def main():
    log_delta_k_steps = np.arange(-6, 0, 0.0606)

    with open('results/sims_%s.csv' % datetime.datetime.now().strftime('%Y%m%d%H%M%S'), 'w') as f:

        f.writelines(",".join(['delta_k','alpha_s','alpha_l','skew','\n']))

        for log_delta_k in log_delta_k_steps:

            delta_k = pow(10, log_delta_k)
            print 'delta_k:',delta_k

            for i in range(10):
                sim = TurnoverModel(1000000, 0.9999, delta_k, 'wake')
                alpha_s = sim.calced_alpha_s.get()
                alpha_l = sim.calced_alpha_l.get()
                skew = sim.calced_skew.get()

                f.writelines(",".join([str(delta_k),str(alpha_s),str(alpha_l),str(skew),'\n']))

        for log_delta_k in log_delta_k_steps:

            delta_k = pow(10, log_delta_k)
            print 'delta_k:',delta_k

            for i in range(10):
                sim = TurnoverModel(300000, 0.9999, delta_k, 'sleep')
                alpha_s = sim.calced_alpha_s.get()
                alpha_l = sim.calced_alpha_l.get()
                skew = sim.calced_skew.get()

                f.writelines(",".join([str(delta_k),str(alpha_s),str(alpha_l),str(skew),'\n']))



if __name__ == '__main__':
    main()
