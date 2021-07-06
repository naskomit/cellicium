from functools import wraps
import multiprocessing as mp
from tqdm.notebook import tqdm

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
    def __init__(self, func, success_cb):
        self.func = func
        self.success_cb = success_cb

    def apply(self, iterable):
        pool = mp.Pool(mp.cpu_count())
        pbar = tqdm()
        pbar.reset(total = len(iterable))
        try:
            for i, res in enumerate(pool.imap(self.func, iterable)):
                if isinstance(res, SubprocessFailure):
                    res.args = list(res.args)
                    res.args[0] = res.args[0] + "\nRemote traceback:\n" + res.traceback
                    raise res.error_type(*res.args)#.with_traceback(res.traceback)
                else:
                    pbar.update()
                    self.success_cb({"value": res.value, "index": i})
        finally:
            pool.close()
        pbar.refresh()

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
