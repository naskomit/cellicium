from functools import wraps
import multiprocessing as mp
from tqdm.notebook import tqdm
import sys
import traceback


class SubprocessSuccess():
    def __init__(self, value):
        self.value = value


class SubprocessFailure():
    def __init__(self, exc_info):
        (error_type, error_instance, tbk) = exc_info
        self.error_type = error_type
        self.args = error_instance.args
        self.traceback = "\n".join(traceback.extract_tb(tbk).format())


class MPRunner():
    def __init__(self, func, success_cb = None, num_workers : int = None, progress = False):
        self.func = func
        self.success_cb = success_cb
        self.progress = progress
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers

    def apply(self, iterable):
        pool = mp.Pool(self.num_workers)
        if self.progress:
            pbar = tqdm(total = len(iterable))
        #pbar.reset(total = len(iterable))
        try:
            for i, res in enumerate(pool.imap(self.func, iterable)):
                if isinstance(res, SubprocessFailure):
                    res.args = list(res.args)
                    res.args[0] = res.args[0] + "\nRemote traceback:\n" + res.traceback
                    raise res.error_type(*res.args)#.with_traceback(res.traceback)
                else:
                    if self.progress:
                        pbar.update()
                    if self.success_cb:
                        self.success_cb({"value": res.value, "index": i})
        finally:
            pool.close()
            if self.progress:
                pbar.close()


def exception2either(func):
    @wraps(func)
    def wrapper(x):
        try:
            res = func(x)
            return SubprocessSuccess(res)
        except Exception:
            ex_info = sys.exc_info()
            return SubprocessFailure(ex_info)
    return wrapper
