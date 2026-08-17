# -*- coding: utf-8 -*-

"""
utils.tracker
Framework for tracking solutions across parameter space
"""

import numpy as np
from typing import Callable, Optional, Union


def const_step_adjuster(step_success, step, *args, **kwargs):
    return step


class SolutionTracker:

    def __init__(self, 
            par_range: np.ndarray, 
            solve_par: Callable,
            criterion: Callable,
            step_0: Optional[float] = None,
            step_adjuster: Callable = const_step_adjuster,
            max_iter: int = 1,
            stop_when_fail: bool = False,
            callback: Optional[Callable] = None
        ):
        self.par_range = par_range
        self.solver = solve_par
        self.criterion = criterion
        self.step_0 = step_0 if step_0 is not None else (self.par_range[-1] - self.par_range[0])/100
        self.adjuster = step_adjuster
        self.max_iter = max_iter
        self.stop_when_fail = stop_when_fail
        self.callback = callback if callback is not None else (lambda *args, **kwargs: None)
        self.reset()

    def reset(self):
        self._par = self.par_range[0]
        self._step = self.step_0
        self._state = self.solver(self._par)
        self._success = True
        self.callback(self._state, self._par, self._step)
    
    def step(self):
        for _ in range(self.max_iter):
            par_next = self._par + self._step
            state_next = self.solver(par_next)
            step_success = self.criterion(self._par, self._state, par_next, state_next)
            self._step = self.adjuster(step_success, self._step, self._par, self._state, par_next, state_next)
            if step_success:
                break
        
        if not step_success:
            self._success = False
            if self.stop_when_fail:
                return
        
        self._state = state_next
        self._par = par_next
        self.callback(self._state, self._par, self._step)
    
    def track(self):
        while (self._par < self.par_range[1]):
            if (self._par + self._step) > self.par_range[1]:
                self._step = self.par_range[1] - self._par
            if self.stop_when_fail and (not self._success):
                break
            self.step()



