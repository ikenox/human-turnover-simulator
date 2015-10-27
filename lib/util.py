from datetime import datetime
import os


def save_and_close_plt(plt, filename):
    str_date = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(os.path.dirname(__file__)+"/../graph/%s_%s.png" % (filename, str_date), bbox_inches='tight')
    plt.clf()
