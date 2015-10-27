from math import log10
import random

from scipy.stats import skew,gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

from lib.util import save_and_close_plt


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

    def __simulate(self):

        # initialize
        self.steps = np.arange(0, self.step - 1)
        k = 0

        if self.stage == 'wake':
            y_start = -1
            before_state = 1
        else:
            x_start = -1
            before_state = 0

        rx = rand_r()
        ry = rand_r()
        self.initialize_records()

        for i in self.steps:
            """
            state 0:x 1:y
            """

            rand = rand_p()
            w = np.exp(-k)
            rxw = pow(rx, w)
            if (self.p >= rand):
                if rxw > ry:
                    state = 0
                else:
                    state = 1
            else:
                if self.record_step_chart:
                    self.random_choices.append(i)
                state = random.getrandbits(1)

            if self.record_step_chart:
                self.states.append(state)
                self.rx_history.append(rx)
                self.rxw_history.append(rxw)
                self.ry_history.append(ry)
                self.k_history.append(k)

            if before_state ^ state:
                if self.stage == 'sleep':
                    if before_state:
                        x_start = i
                        self.turnover_angles.append(rx * (i-y_start))
                        self.turnover_intervals.append(interval)
                    else:
                        y_start = i
                        interval = i - x_start

                if self.stage == 'wake':
                    if before_state:
                        x_start = i
                        interval = i - y_start
                    else:
                        y_start = i
                        self.turnover_intervals.append(interval)
                        self.turnover_angles.append(ry * (i-x_start))

            if state:  # state=y
                ry = rand_r()
                k = 0
            else:  # state=x
                rx = rand_r()
                k += self.delta_k

            before_state = state


        interval_sum = 0
        for interval in self.turnover_intervals:
            interval_sum += interval
            self.turnover_times.append(interval_sum)



    def initialize_records(self):

        self.turnover_times = []
        self.turnover_angles = []
        self.turnover_intervals = []

        self.calced_log_turnover_intervals = self.CalcedAttr(self.__calc_log_results)
        self.calced_log_turnover_angles = self.CalcedAttr(self.__calc_log_results)
        self.calced_fn_series = self.CalcedAttr(self.__calc_fluctuations)
        self.calced_alpha_series = self.CalcedAttr(self.__calc_fluctuations)
        self.calced_alpha_s = self.CalcedAttr(self.__calc_fluctuations)
        self.calced_alpha_l = self.CalcedAttr(self.__calc_fluctuations)
        self.calced_skew = self.CalcedAttr(self.__calc_skew)

        if self.record_step_chart:
            self.rx_history = []
            self.rxw_history = []
            self.ry_history = []
            self.k_history = []
            self.states = []
            self.random_choices = []

    def __init__(self, step, p, delta_k, stage, record_step_chart=False):
        self.step = step
        self.p = p
        self.delta_k = delta_k
        self.stage = stage
        self.record_step_chart = record_step_chart

        self.initialize_records()

        self.__simulate()

    def __calc_fluctuations(self):
        """
        Calculating scaling exponent of F(n), alpha(n), alpha_s and alpha_l
        """

        log_xsteps = np.linspace(0.5, max(self.calced_log_turnover_intervals.get())-0.5, 15)
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

        self.calced_fn_series.set(fn_series)
        self.calced_alpha_series.set(alpha_series)
        self.calced_alpha_l.set(alpha_l)
        self.calced_alpha_s.set(alpha_s)

        return fn_series, alpha_series, alpha_s, alpha_l

    def __calc_log_results(self):
        self.calced_log_turnover_intervals.set([log10(i) for i in self.turnover_intervals])
        self.calced_log_turnover_angles.set([log10(i) for i in self.turnover_angles])

    def __calc_skew(self):
        self.calced_skew.set(skew([log10(i) for i in self.turnover_intervals]))

    def save_interval_angle_dist(self):
        """
        Plot 2D probability distribution P(tau,a)
        """

        plt.figure(figsize=(3, 4))

        xmin = min(self.calced_log_turnover_intervals.get())
        xmax = max(self.calced_log_turnover_intervals.get())
        ymin = min(self.calced_log_turnover_angles.get())
        ymax = max(self.calced_log_turnover_angles.get())

        xx = np.linspace(-1, 6, 32)
        yy = np.linspace(-2, 3, 32)
        X, Y = np.meshgrid(xx, yy)
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([self.calced_log_turnover_intervals.get(), self.calced_log_turnover_angles.get()])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions), X.shape)
        plt.imshow(np.rot90(Z.T), cmap=plt.cm.gist_earth_r, extent=[-1, 6, -2, 3], aspect=2)

        plt.xlabel(r'$\log_{10}(\tau)$')
        plt.ylabel(r'$\log_{10}(A)$')

        plt.plot(self.calced_log_turnover_intervals.get(), self.calced_log_turnover_angles.get(), 'k.', markersize=5, c='#FF0000')
        save_and_close_plt(plt, 'interval_angle_dist')

    def save_interval_ccdf(self):
        """
        Plot complementary cumulative distribution function P(x>=tau)
        """
        plt.figure(figsize=(3, 4))

        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$P(x\geq\tau)$')

        slis = sorted(self.calced_log_turnover_intervals.get())
        l = len(slis)
        i = 0
        y = []
        for x in slis:
            y.append(1.0 * (l - i) / l)
            i += 1

        plt.scatter(slis, y, s=100, c='white')
        save_and_close_plt(plt, 'interval_ccdf')

    def save_fluctuation(self):
        """
        Plot F(n) and alpha(n)
        """
        plt.figure(figsize=(3, 4))

        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.005, 100)
        plt.scatter(self.calced_fn_series.get()[0], self.calced_fn_series.get()[1], s=100, c='#999999')
        plt.xlabel(r'$n$')
        plt.ylabel(r'$F(n)$')
        ax = plt.twinx()
        ax.set_ylim(0, 1)
        ax.set_ylabel(r'$\alpha(n)$')
        ax.scatter(self.calced_alpha_series.get()[0], self.calced_alpha_series.get()[1], s=100, c='white')
        save_and_close_plt(plt, 'fn_and_alpha')

    def save_step_chart(self):
        if self.record_step_chart:
            plt.figure(figsize=(20, 12))

            plt.subplot(511)
            plt.title(r"State (0:state$x$, 1:state$y$)")
            plt.ylim(-0.15, 1.15)
            plt.step(self.steps, self.states, color="#0E7AC4")

            plt.subplot(512)
            plt.title(r"$R_x^W$")
            plt.ylim(-0.05, 1.05)
            plt.step(self.steps, self.rxw_history, color="#0E7AC4")

            plt.subplot(513)
            plt.title(r"$R_y$")
            plt.ylim(-0.05, 1.05)
            plt.step(self.steps, self.ry_history, color="#0E7AC4")

            plt.subplot(514)
            plt.title(r"$K$")
            plt_line(plt, self.random_choices, color='green')
            plt.step(self.steps, self.k_history, color="#0E7AC4")

            plt.subplot(515)
            plt.title(r"Turnover angle $A$")
            plt_line(plt, self.turnover_times)
            plt.scatter(self.turnover_times, self.turnover_angles, color="#0E7AC4", s=3)

            plt.tight_layout()
            save_and_close_plt(plt, 'walking')
        else:
            raise Exception('Walking chart is not recorded.')

def plt_line(plt, x_series, color='red'):
    for t in x_series:
        plt.axvline(x=t, color=color, linewidth=0.5)


def rand_r():
    r = np.random.random()
    return r


def rand_p():
    r = np.random.random()
    return r
