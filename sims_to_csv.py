import datetime
import multiprocessing as mp

import numpy as np

from lib.simulate import TurnoverModel

def main():
    proc = 10

    log_delta_k_steps = np.arange(-6, 0, 0.0606)

    with open('results/sims_%s.csv' % datetime.datetime.now().strftime('%Y%m%d%H%M%S'), 'w') as f:

        pool = mp.Pool(proc)

        f.writelines(",".join(['stage', 'delta_k','alpha_s','alpha_l','skew','\n']))

        print "=== Sleep stage ==="

        for log_delta_k in log_delta_k_steps:
            delta_k = pow(10, log_delta_k)
            print 'delta_k:',delta_k
            results = pool.map(sim, [(300000, delta_k, 'sleep')]*proc)
            for row in results:
                f.writelines(",".join(['sleep',str(delta_k),str(row[0]),str(row[1]),str(row[2]),'\n']))

        print "=== Wake stage ==="

        for log_delta_k in log_delta_k_steps:
            delta_k = pow(10, log_delta_k)
            print 'delta_k:',delta_k
            results = pool.map(sim, [(3000000, delta_k, 'wake')]*proc)
            for row in results:
                f.writelines(",".join(['wake',str(delta_k),str(row[0]),str(row[1]),str(row[2]),'\n']))

def sim((step_num, delta_k, stage)):
    sim = TurnoverModel(step_num, 0.9999, delta_k, stage)
    alpha_s = sim.calced_alpha_s.get()
    alpha_l = sim.calced_alpha_l.get()
    skew = sim.calced_skew.get()

    return alpha_s,alpha_l,skew

if __name__ == '__main__':
    main()
