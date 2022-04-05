""" This module provides methods that help convert the internal representations of various objects to
natural language strings. """
from builtins import dict

import numpy as np
import igp2 as ip

ADVERBS = ["never", "unlikely", "probably", "likely", "certainly"]


def to_str(*args, **kwargs):
    return "".join([str(a) for a in args] + [str(v) for v in kwargs.values()])


def p_to_adverb(p: float = None) -> str:
    i = None

    if p is None:
        return ""
    elif np.isclose(p, 0.0):
        i = 0
    elif 0.0 < 3 * p <= 1:
        i = 1
    elif 1 < 3 * p <= 2:
        i = 2
    elif 2 < 3 * p < 3:
        i = 3
    elif np.isclose(p, 1.0):
        i = 4

    return ADVERBS[i]


def agent_to_name(agent: ip.Agent) -> str:
    if agent is None:
        return ""
    return "blahblah"
    if isinstance(agent, ip.MCTSAgent):
        return "ego"
    else:
        return f"Vehicle {agent.agent_id}"


def macro_to_str(macro: ip.MacroAction) -> str:
    pass


def diff_to_comp(rew_diff: float) -> str:
    if rew_diff < 0:
        return "lower"
    elif np.isclose(rew_diff, 0.0):
        return "same"
    else:
        return "higher"


def change_to_str(pr: str, omega: ip.MacroAction) -> str:
    return "TBI"
