from datetime import datetime


def save_and_close_plt(plt, filename):
    str_date = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig("graph/%s_%s.png" % (filename, str_date), bbox_inches='tight')
    plt.clf()
