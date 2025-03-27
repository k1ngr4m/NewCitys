import logging
import time
import os
from datetime import datetime, timezone, timedelta

# 控制台输出
STREAM = True
# 获取文件的绝对路径
abs_path = os.path.abspath(__file__)
# print(abs_path)

# 获取文件所在目录的上一级目录，也就是根目录
project_path = os.path.dirname(os.path.dirname(abs_path))
# print(project_path)

# 获取log日志目录的全路径
_log_path = project_path + os.sep + "log"

# 返回日志目录
def get_log_path():
    return _log_path

class UTC8Formatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, timezone(timedelta(hours=8)))
        return dt

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            t = dt.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s


class LogUtil:
    def __init__(self):
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            self.log_name = '{}.log'.format(time.strftime("%Y_%m_%d", time.localtime()))
            self.log_path_file = os.path.join(get_log_path(), self.log_name)
            fh = logging.FileHandler(self.log_path_file, encoding='utf-8', mode='a')
            fh.setLevel(logging.DEBUG)
            formatter = UTC8Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            if STREAM:
                fh_stream = logging.StreamHandler()
                fh_stream.setLevel(logging.DEBUG)
                fh_stream.setFormatter(formatter)
                self.logger.addHandler(fh_stream)

    def log(self):
        # 返回定义好的logger对象
        return self.logger


logger = LogUtil().log()

if __name__ == '__main__':
    logger.error('test')