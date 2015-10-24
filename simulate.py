from datetime import datetime
from math import log10
import random

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew


#
# @return turnover_intervals, turnover_angles
#
def simulate_turnover(step, p, delta_k, stage, gen_walking_chart=False):
    # initialize
    steps = np.arange(0, step - 1)
    k = 0
    rx = rand_not0()
    ry = rand_not0()
    before_state = 'x'

    turnover_times = []
    turnover_angles = []
    turnover_intervals = []

    if gen_walking_chart:
        rx_history = []
        rxw_history = []
        ry_history = []
        k_history = []
        states = []
        random_choices = []

    for i in steps:

        rand = rand_not0()
        w = np.exp(-k)
        rxw = pow(rx, w)
        if (p >= rand):
            if rxw > ry:
                state = 'x'
            else:
                state = 'y'
        else:
            if gen_walking_chart:
                random_choices.append(i)

            if random.getrandbits(1):
                state = 'x'
            else:
                state = 'y'

        if gen_walking_chart:
            states.append(state)
            rx_history.append(rx)
            rxw_history.append(rxw)
            ry_history.append(ry)
            k_history.append(k)

        if state == 'y':  # state=y
            ry = rand_not0()
            k = 0
        elif state == 'x':  # state=x
            rx = rand_not0()
            k += delta_k
        else:
            print "error: invalid state"
            exit()

        if before_state == 'y' and state == 'x':
            if len(turnover_times) > 0:
                interval = i - turnover_times[-1]
            else:
                interval = i

            turnover_angles.append(rx * (interval))
            turnover_intervals.append(interval)
            turnover_times.append(i)

        before_state = state

    if gen_walking_chart:
        plt.figure(figsize=(20,12))

        plt.subplot(511)
        plt.title(r"State")
        plt.ylim(-0.15,1.15)
        plt.step(steps, states, color="#0E7AC4")

        plt.subplot(512)
        plt.title(r"$R_x^W$")
        plt.ylim(-0.05,1.05)
        plt.step(steps, rxw_history, color="#0E7AC4")

        plt.subplot(513)
        plt.title(r"$R_y$")
        plt.ylim(-0.05,1.05)
        plt.step(steps, ry_history, color="#0E7AC4")

        plt.subplot(514)
        plt.title(r"$K$")
        plt_line(plt, random_choices, color='green')
        plt.step(steps, k_history, color="#0E7AC4")

        plt.subplot(515)
        plt.title(r"$Turnover angle$")
        plt_line(plt, turnover_times)
        plt.scatter(turnover_times, turnover_angles,color="#0E7AC4", s=3)

        plt.tight_layout()

        save_and_close_plt(plt,'walking')

    return turnover_intervals, turnover_angles


def save_and_close_plt(plt, filename):
    str_date = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig("graph/%s_%s.png" % (filename, str_date))
    plt.clf()


def save_interval_angle_dist(turnover_intervals, turnover_angles):

    log_turnover_intervals = [log10(i) for i in turnover_intervals]
    log_turnover_angles = [log10(i) for i in turnover_angles]
    xmin = min(log_turnover_intervals)
    xmax = max(log_turnover_intervals)
    ymin = min(log_turnover_angles)
    ymax = max(log_turnover_angles)

    # ========
    # plt scatter and 2-D dist
    # ========
    plt.figure()
    xx = np.linspace(xmin - 1, xmax + 1, 32)
    yy = np.linspace(ymin - 1, ymax + 1, 32)
    X, Y = np.meshgrid(xx, yy)
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([log_turnover_intervals, log_turnover_angles])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions), X.shape)
    plt.imshow(np.rot90(Z.T), cmap=plt.cm.gist_earth_r, extent=[xmin - 1, xmax + 1, ymin - 1, ymax + 1])

    plt.plot(log_turnover_intervals, log_turnover_angles, 'k.', markersize=5, c='#FF0000')
    save_and_close_plt(plt, 'interval_angle_dist')


def main():
    # param
    p = 0.9999
    delta_k = 0.000001
    step = 300000

    # initialize
    steps = np.arange(0, step - 1)
    k = 0
    rx = rand_not0()
    ry = rand_not0()
    before_state = 'x'

    rx_history = []
    rxw_history = []
    ry_history = []
    k_history = []
    states = []
    random_choices = []
    turnover_times = []
    turnover_angles = []
    turnover_intervals = []

    for i in steps:

        rand = rand_not0()
        w = np.exp(-k)
        rxw = pow(rx, w)
        if (p >= rand):
            if rxw > ry:
                state = 'x'
            else:
                state = 'y'
        else:
            random_choices.append(i)

            if random.getrandbits(1):
                state = 'x'
            else:
                state = 'y'

        # states.append(state)
        # rx_history.append(rx)
        # rxw_history.append(rxw)
        # ry_history.append(ry)
        # k_history.append(k)

        if state == 'y':  # state=y
            ry = rand_not0()
            k = 0
        elif state == 'x':  # state=x
            rx = rand_not0()
            k += delta_k
        else:
            print "error: invalid state"
            exit()

        if before_state == 'y' and state == 'x':
            if len(turnover_times) > 0:
                interval = i - turnover_times[-1]
            else:
                interval = i

            turnover_angles.append(rx * (interval))
            turnover_intervals.append(interval)
            turnover_times.append(i)

        before_state = state

    # calc values


    # plot

    str_date = datetime.now().strftime('%Y%m%d%H%M%S')

    # plt.figure(figsize=(20,12))
    #
    # plt.subplot(511)
    # plt.title(r"State")
    # plt.ylim(-0.15,1.15)
    # plt.step(steps, states, color="#0E7AC4")
    #
    # plt.subplot(512)
    # plt.title(r"$R_x^W$")
    # plt.ylim(-0.05,1.05)
    # plt.step(steps, rxw_history, color="#0E7AC4")
    #
    # plt.subplot(513)
    # plt.title(r"$R_y$")
    # plt.ylim(-0.05,1.05)
    # plt.step(steps, ry_history, color="#0E7AC4")
    #
    # plt.subplot(514)
    # plt.title(r"$K$")
    # plt_line(plt, random_choices, color='green')
    # plt.step(steps, k_history, color="#0E7AC4")
    #
    # plt.subplot(515)
    # plt.title(r"$Turnover angle$")
    # plt.xlim(None,step)
    # plt_line(plt, turnover_times)
    # plt.scatter(turnover_times, turnover_angles,color="#0E7AC4", s=3)
    #
    # plt.tight_layout()
    # plt.savefig("graph/result_%s.png" % str_date)
    # plt.clf()
    #

    log_turnover_intervals = [log10(i) for i in turnover_intervals]
    log_turnover_angles = [log10(i) for i in turnover_angles]
    xmin = min(log_turnover_intervals)
    xmax = max(log_turnover_intervals)
    ymin = min(log_turnover_angles)
    ymax = max(log_turnover_angles)

    # ========
    # plt scatter and 2-D dist
    # ========
    plt.figure()
    xx = np.linspace(xmin - 1, xmax + 1, 32)
    yy = np.linspace(ymin - 1, ymax + 1, 32)
    X, Y = np.meshgrid(xx, yy)
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([log_turnover_intervals, log_turnover_angles])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions), X.shape)
    plt.imshow(np.rot90(Z.T), cmap=plt.cm.gist_earth_r, extent=[xmin - 1, xmax + 1, ymin - 1, ymax + 1])

    plt.plot(log_turnover_intervals, log_turnover_angles, 'k.', markersize=5, c='#FF0000')
    plt.savefig("graph/scatter_%s.png" % str_date)
    plt.clf()

    # ========
    # plt ccdf
    # ========
    plt.figure(figsize=(3, 4))

    slis = sorted(log_turnover_intervals)
    l = len(slis)
    i = 0
    y = []
    for x in slis:
        y.append(1.0 * (l - i) / l)
        i += 1

    plt.scatter(slis, y, s=100)
    plt.savefig("graph/ccdf_%s.png" % str_date)
    plt.clf()

    print skew(slis)


    # ========
    # plt alpha and skew
    # ========
    xx = np.linspace(0.6, 3.8, 20)
    xx_line = [int(pow(10, x)) for x in xx]
    ave_wn = []
    ave_wn2 = []
    fnlist = []

    sum = 0
    for x in xx_line:

        changepoints = []

        for t in turnover_times:
            imin = t - x
            imax = t
            if t - x < 0:
                imin = 0
            if t > step - 1 - x:
                imax = step - 1 - x

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
                raise Exception

            valueranges.append({'range': (changepoints[i][0], changepoints[i + 1][0]), 'value': tmp_value})

        window_num = step - 1 - x

        wnsum = 0
        wn2sum = 0

        for vr in valueranges:
            wnsum += (vr['range'][1] - vr['range'][0]) * vr['value']
            wn2sum += (vr['range'][1] - vr['range'][0]) * pow(vr['value'], 2)

        ave_wn = 1.0 * wnsum / window_num
        ave_wn2 = (1.0 * wn2sum / window_num)

        fnlist.append(log10(np.sqrt(ave_wn2 - ave_wn ** 2)))

    plt.figure(figsize=(3, 4))
    plt.ylim(-2.1, 2)

    plt.scatter(xx, fnlist, s=100)

    yy_a = []
    xx_a = []

    for i in range(len(fnlist[:-1])):
        delx = (xx[i + 1] - xx[i])
        xx_a.append(xx[i] + delx / 2)
        yy_a.append((fnlist[i + 1] - fnlist[i]) / delx)

    plt.scatter(xx_a, yy_a, s=100)
    plt.savefig("graph/f_a_%s.png" % str_date)
    plt.clf()

    alphas = zip(xx_a, yy_a)

    alphas_s = [al[1] for al in alphas if al[0] <= 2]
    alphas_l = [al[1] for al in alphas if al[0] >= 2]

    print np.average(alphas_s), np.average(alphas_l)


def plt_line(plt, x_series, color='red'):
    for t in x_series:
        plt.axvline(x=t, color=color, linewidth=0.5)


def rand_not0():
    r = np.random.random()
    return r


def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1 + m2, m1 - m2


if __name__ == '__main__':
    main()
