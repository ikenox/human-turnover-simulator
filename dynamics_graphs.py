from math import log10
import argparse

from scipy.stats import skew

from lib.simulate import TurnoverModel


def main():

    parser = argparse.ArgumentParser(description='Simulate and plot graphs of human turnover steps.')
    parser.add_argument('step', type=int,
                       help='an integer for the accumulator')
    parser.add_argument('p', type=float,
                       help='an integer for the accumulator')
    parser.add_argument('K', type=float,
                       help='an integer for the accumulator')
    parser.add_argument('stage', type=str,
                       help='wake or sleep')
    parser.add_argument('--stepchart', default=False, action="store_true",
                        help='if record and plot step chart')

    args = parser.parse_args()

    sim = TurnoverModel(args.step, args.p, args.K, args.stage, record_step_chart=args.stepchart)
    sim.save_fluctuation()
    sim.save_interval_angle_dist()
    sim.save_interval_ccdf()
    if args.stepchart:
        sim.save_step_chart()

    print '===== turnover intervals ====='
    print sim.turnover_intervals
    print '===== turnover times,intervals,angles ====='
    print zip(sim.turnover_times,sim.turnover_intervals,sim.turnover_angles)
    print '===== alpha s,l ====='
    print sim.calced_alpha_s.get(), sim.calced_alpha_l.get()
    print '===== skew of log10(tau)====='
    print skew([log10(i) for i in sim.turnover_intervals])


if __name__ == '__main__':
    main()