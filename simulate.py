from datetime import datetime
from math import log10
import random

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


class TurnoverModel:

    class CalcedAttr:
        """
        An attribute which can be calculated from attribute.
        CalcedAttr reduces redundant calculation
        """

        def __init__(self, calc_func):
            self.value = None
            self.calc_func = calc_func

        def get(self):
            if self.value:
                return self.value
            else:
                self.calc_func()
                return self.value

        def set(self, value):
            self.value = value

    def __init__(self, step, p, delta_k, stage, gen_walking_chart=False):
        self.step = step
        self.p = p
        self.delta_k = delta_k
        self.stage = stage
        self.gen_walking_chart = gen_walking_chart

        self.log_turnover_intervals = self.CalcedAttr(self.calc_log_results)
        self.log_turnover_angles = self.CalcedAttr(self.calc_log_results)
        self.fn_series = self.CalcedAttr(self.calc_fluctuations)
        self.alpha_series = self.CalcedAttr(self.calc_fluctuations)
        self.alpha_s = self.CalcedAttr(self.calc_fluctuations)
        self.alpha_l = self.CalcedAttr(self.calc_fluctuations)

        """
        Simulate
        """
        # initialize
        steps = np.arange(0, self.step - 1)
        k = 0
        rx = rand_r()
        ry = rand_r()
        before_state = 0

        self.turnover_times = []
        self.turnover_angles = []
        self.turnover_intervals = []

        if self.gen_walking_chart:
            rx_history = []
            rxw_history = []
            ry_history = []
            k_history = []
            states = []
            random_choices = []

        for i in steps:
            """
            state 0:x 1:y
            """

            rand = rand_p()
            w = np.exp(-k)
            rxw = pow(rx, w)
            if (p >= rand):
                if rxw > ry:
                    state = 0
                else:
                    state = 1
            else:
                if self.gen_walking_chart:
                    random_choices.append(i)
                state = random.getrandbits(1)

            if self.gen_walking_chart:
                states.append(state)
                rx_history.append(rx)
                rxw_history.append(rxw)
                ry_history.append(ry)
                k_history.append(k)

            if state:  # state=y
                ry = rand_r()
                k = 0
            else:  # state=x
                rx = rand_r()
                k += self.delta_k

            if before_state and not state:
                if len(self.turnover_times) > 0:
                    interval = i - self.turnover_times[-1]
                else:
                    interval = i

                self.turnover_angles.append(rx * (interval))
                self.turnover_intervals.append(interval)
                self.turnover_times.append(i - 1)

            before_state = state

        if before_state:
            if len(self.turnover_times) > 0:
                interval = i - self.turnover_times[-1]
            else:
                interval = i
            self.turnover_angles.append(rx * (interval))
            self.turnover_intervals.append(interval)
            self.turnover_times.append(i - 1)

        if self.gen_walking_chart:
            plt.figure(figsize=(20, 12))

            plt.subplot(511)
            plt.title(r"State")
            plt.ylim(-0.15, 1.15)
            plt.step(steps, states, color="#0E7AC4")

            plt.subplot(512)
            plt.title(r"$R_x^W$")
            plt.ylim(-0.05, 1.05)
            plt.step(steps, rxw_history, color="#0E7AC4")

            plt.subplot(513)
            plt.title(r"$R_y$")
            plt.ylim(-0.05, 1.05)
            plt.step(steps, ry_history, color="#0E7AC4")

            plt.subplot(514)
            plt.title(r"$K$")
            plt_line(plt, random_choices, color='green')
            plt.step(steps, k_history, color="#0E7AC4")

            plt.subplot(515)
            plt.title(r"$Turnover angle$")
            plt_line(plt, self.turnover_times)
            plt.scatter(self.turnover_times, self.turnover_angles, color="#0E7AC4", s=3)

            plt.tight_layout()
            save_and_close_plt(plt, 'walking')

    def calc_fluctuations(self):

        """
        Calculating scaling exponent of F(n), alpha(n), alpha_s and alpha_l
        """

        log_xsteps = np.linspace(0.1, 4.5, 15)
        line_xsteps = [int(pow(10, x)) for x in log_xsteps]
        fn_list = []

        for x in line_xsteps:

            changepoints = []

            for t in self.turnover_times:
                imin = t - x
                imax = t
                if t - x < 0:
                    imin = 0
                if t > self.step - 1 - x:
                    imax = self.step - 1 - x

                changepoints.append((imin, 'from'))
                changepoints.append((imax, 'to'))

            changepoints.sort(cmp=lambda x, y: cmp(x[0], y[0]))

            valueranges = []
            tmp_value = 0
            for i in range(len(changepoints) - 1):
                if changepoints[i][1] == 'from':
                    tmp_value += 1
                elif changepoints[i][1] == 'to':
                    tmp_value -= 1
                else:
                    raise Exception('Unexpected changepoints')

                valueranges.append({'range': (changepoints[i][0], changepoints[i + 1][0]), 'value': tmp_value})

            window_num = self.step - 1 - x

            wnsum = 0
            wn2sum = 0

            for vr in valueranges:
                wnsum += (vr['range'][1] - vr['range'][0]) * vr['value']
                wn2sum += (vr['range'][1] - vr['range'][0]) * pow(vr['value'], 2)

            ave_wn = 1.0 * wnsum / window_num
            ave_wn2 = (1.0 * wn2sum / window_num)

            fn_list.append(np.sqrt(ave_wn2 - ave_wn ** 2))

        fn_series = (line_xsteps, fn_list)

        log_fn_list = [log10(fn) for fn in fn_list]

        alpha_series = ([], [])
        for i in range(len(fn_list[:-1])):
            log_delx = (log_xsteps[i + 1] - log_xsteps[i])
            alpha_series[0].append(np.power(10, log_xsteps[i] + log_delx / 2))
            alpha_series[1].append((log_fn_list[i + 1] - log_fn_list[i]) / log_delx)

        alpha_s = np.average([al for i, al in enumerate(alpha_series[1]) if log10(alpha_series[0][i]) <= 2])
        alpha_l = np.average([al for i, al in enumerate(alpha_series[1]) if log10(alpha_series[0][i]) >= 2])

        self.fn_series.set(fn_series)
        self.alpha_series.set(alpha_series)
        self.alpha_l.set(alpha_l)
        self.alpha_s.set(alpha_s)

        return fn_series, alpha_series, alpha_s, alpha_l

    def calc_log_results(self):
        self.log_turnover_intervals.set([log10(i) for i in self.turnover_intervals])
        self.log_turnover_angles.set([log10(i) for i in self.turnover_angles])

    def save_interval_angle_dist(self):
        """
        2-dimention probability distribution P(tau,a)
        """

        xmin = min(self.log_turnover_intervals)
        xmax = max(self.log_turnover_intervals)
        ymin = min(self.log_turnover_angles)
        ymax = max(self.log_turnover_angles)

        plt.figure()
        xx = np.linspace(xmin - 1, xmax + 1, 32)
        yy = np.linspace(ymin - 1, ymax + 1, 32)
        X, Y = np.meshgrid(xx, yy)
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([self.log_turnover_intervals, self.log_turnover_angles])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions), X.shape)
        plt.imshow(np.rot90(Z.T), cmap=plt.cm.gist_earth_r, extent=[xmin - 1, xmax + 1, ymin - 1, ymax + 1])

        plt.plot(self.log_turnover_intervals, self.log_turnover_angles, 'k.', markersize=5, c='#FF0000')
        save_and_close_plt(plt, 'interval_angle_dist')

    def save_interval_ccdf(self):
        """
        Complementary cumulative distribution function P(x>=tau)
        """
        plt.figure(figsize=(3, 4))

        slis = sorted(self.log_turnover_intervals)
        l = len(slis)
        i = 0
        y = []
        for x in slis:
            y.append(1.0 * (l - i) / l)
            i += 1

        plt.scatter(slis, y, s=100)
        save_and_close_plt(plt, 'interval_ccdf')

    def save_fluctuation(self):
        """
        F(n) and alpha(n)
        """
        plt.figure(figsize=(3, 4))

        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.005, 100)
        plt.scatter(self.fn_series[0], self.fn_series[1], s=100)
        ax = plt.twinx()
        ax.set_ylim(0, 1)
        ax.scatter(self.alpha_series[0], self.alpha_series[1], s=100)
        save_and_close_plt(plt, 'fn_and_alpha')


def plt_line(plt, x_series, color='red'):
    for t in x_series:
        plt.axvline(x=t, color=color, linewidth=0.5)


def save_and_close_plt(plt, filename):
    str_date = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig("graph/%s_%s.png" % (filename, str_date))
    plt.clf()


def rand_r():
    r = np.random.random()
    return r


def rand_p():
    r = np.random.random()
    return r
