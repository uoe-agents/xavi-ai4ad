""" This module provides methods that help convert the internal representations of various objects to
natural language strings. """
import random
from typing import Union, List, Optional
from dataclasses import dataclass

import numpy as np
import igp2 as ip

ADVERBS = ["never", "unlikely", "probably", "likely", "certainly"]


@dataclass
class Counterfactual:
    omegas: List[ip.MCTSAction]
    outcome: str
    p_outcome: float


@dataclass
class Effect:
    relation: Union[float, ip.Agent]
    reward: Optional[str]


@dataclass
class Cause:
    agent: ip.Agent


@dataclass
class Property:
    pass


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
    elif isinstance(agent, ip.MCTSAgent):
        return "ego"
    else:
        return f"Vehicle {agent.agent_id}"


def macro_to_str(agent_id, frame, scenario_map, macro: ip.MCTSAction) -> str:
    ma = macro.macro_action_type(agent_id=agent_id,
                                 frame=frame,
                                 scenario_map=scenario_map,
                                 open_loop=True,
                                 **macro.ma_args)
    if isinstance(ma, ip.Continue):
        return f"{random.choice(['gone', 'driven', 'continued'])} straight"
    elif isinstance(ma, ip.Exit):
        straight_threshold = 1e-2
        direction = "left" if ma.orientation < -straight_threshold \
            else "right" if ma.orientation > straight_threshold \
            else "straight"
        if direction == "straight":
            return f"{random.choice(['gone', 'driven', 'continued'])} straight"
        else:
            return f"turned {direction}"
    elif isinstance(ma, ip.ChangeLane):
        direction = "left" if ma.left else "right"
        return f"{random.choice(['changed', 'switched', 'moved'])} lanes to the {direction}"
    else:
        return str(ma)


def change_to_str(pr: str, omega: ip.MacroAction) -> str:
    return "TBI"


def reward_to_str(r: str) -> str:
    return {
        "reward_time": "time to goal",
        "reward_jerk": "jerk",
        "reward_angular_acceleration": "angular acceleration",
        "reward_curvature": "curvature",
        None: ""
    }[r]


def outcome_to_str(o: str) -> str:
    return {
        "outcome_done": "reached the goal",
        "outcome_coll": "collided",
        "outcome_dead": "not reached the goal",
        "outcome_term": "not reached the goal",
        None: ""
    }[o]


def diff_to_comp(rew_diff: Union[ip.Agent, float]) -> str:
    if isinstance(rew_diff, ip.Agent):
        return f"Vehicle {rew_diff.agent_id}"
    elif rew_diff is None:
        return ""
    elif rew_diff < 0:
        return "lower"
    elif np.isclose(rew_diff, 0.0):
        return "same"
    else:
        return "higher"


def none(name, **kwargs): return kwargs[name] is None


def is_type(name, t, **kwargs): return not none(name, **kwargs) and isinstance(kwargs[name], t) == 1


def len_eq1(name, **kwargs):
    return not none(name, **kwargs) and \
           (not hasattr(kwargs[name], "__len__") or
            len(kwargs[name]) == 1)


def len_gt1(name, **kwargs):
    return not none(name, **kwargs) and \
           hasattr(kwargs[name], "__len__") and \
           len(kwargs[name]) > 1
