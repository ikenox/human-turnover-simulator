from math import log10
from scipy.stats import skew
from simulate import simulate_turnover, save_fluctuation, save_interval_angle_dist, save_interval_ccdf, calc_fluctuations


def main():

    step = 300000

    turnover_times, turnover_intervals, turnover_angles = simulate_turnover(step, 0.9999, 0.572, 'wake',
                                                                            gen_walking_chart=False)
    log_turnover_intervals = [log10(i) for i in turnover_intervals]
    log_turnover_angles = [log10(i) for i in turnover_angles]

    save_interval_angle_dist(log_turnover_intervals, log_turnover_angles)
    save_interval_ccdf(log_turnover_intervals)

    fn_series, alpha_series, alpha_s, alpha_l = calc_fluctuations(turnover_times, turnover_intervals, step)
    save_fluctuation(fn_series, alpha_series)

    print 'skew:',skew([log10(i) for i in turnover_intervals])


if __name__ == '__main__':
    main()