from typing import Dict, Tuple, List

import numpy as np
import igp2 as ip


def softmax(x, axis: int = 0):
    """ Calculate a numerically stable version of softmax. """
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Sample:
    """ Utility class to group samples of a sampling step in MCTS. """

    def __init__(self, samples: Dict[int, Tuple[ip.GoalWithType, ip.VelocityTrajectory]]):
        self.samples = samples

    def __eq__(self, other: "Sample"):
        if not isinstance(other, Sample):
            return False
        if self == other:
            return True
        for aid, sample in other.samples.items():
            if aid not in self.samples:
                return False
            return self.samples[aid] == sample

    def __getitem__(self, item):
        return self.samples[item]

    @property
    def agents(self) -> List[int]:
        """ The IDs of agents in the sample. """
        return list(self.samples)
