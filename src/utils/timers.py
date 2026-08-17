# -*- coding: utf-8 -*-

import time
from typing import Literal


class ProcTimer:

    def __init__(self, start: bool = False) -> None:
        self._starttime = None
        self._logtimes = None
        self._loginfos = None
        if start:
            self.start()

    def start(self, num: bool = True) -> None:
        self._starttime = time.perf_counter()
        self._logtimes = list((self._starttime,))
        if num:
            self._loginfos = list((0,))
        else:
            self._loginfos = list(('start',))

    def clear(self) -> None:
        self._starttime = None
        self._logtimes = None
        self._loginfos = None

    def flag(self, loginfo=None, print_str: bool = False, **kwargs) -> None:
        self._logtimes.append(time.perf_counter())
        self._loginfos.append(loginfo)
        if print_str:
            self.print_elapse(**kwargs)

    def elapse_time(self, increment: bool = True) -> float:
        if increment:
            return self._logtimes[-1] - self._logtimes[-2]
        else:
            return self._logtimes[-1] - self._logtimes[0]

    def print_elapse(self, mode: Literal['0', '+', '0+'] = '0', **kwargs) -> None:
        t_inc = self.elapse_time(increment=True)
        t_tot = self.elapse_time(increment=False)
        t_info = self._loginfos[-1]
        if mode == '+':
            print("Elapse time (+) = {:8.2f} | Info: {}".format(t_inc, t_info), **kwargs)
        elif mode == '0':
            print("Elapse time (0) = {:8.2f} | Info: {}".format(t_tot, t_info), **kwargs)
        elif mode == '0+':
            print("Elapse time = {:8.2f} ({:+8.2f}) | Info: {}".format(t_tot, t_inc, t_info), **kwargs)

