from typing import Tuple, Dict, List

import igp2 as ip
import numpy as np

from xavi.util import softmax


class XAVINode(ip.Node):
    """ Subclass of ip.Node that allows decomposition of Q-values based on non-ego samples. """

    def __init__(self, key: Tuple, state: Dict[int, ip.AgentState], actions: List[ip.MCTSAction]):
        super(XAVINode, self).__init__(key, state, actions)
        self._selected_q_idx = None

    def expand_samples(self, num_samples: int):
        if self._actions is None:
            raise TypeError("Cannot expand node without actions")
        self._q_values = np.zeros((num_samples, len(self._actions)))
        self._action_visits = np.zeros((num_samples, len(self._actions)), dtype=np.int32)

    def select_q_idx(self, idx: int = 0):
        """ Specify which Q-values to update, and update all children recurisvely nodes as well. """
        self._selected_q_idx = idx
        for child in self.children.values():
            child.select_q_idx(idx)

    def action_probabilities(self, alpha: float = 1.0):
        """ Get the current action probabilities for the selected sampling index.

        Args:
            alpha: Scaling parameter in the softmax.
        """
        probs = softmax(alpha * self._q_values, axis=1)
        return probs[self._selected_q_idx, :]

    @property
    def q_values(self) -> np.ndarray:
        """ Return the currently selected Q-values of the node"""
        return self._q_values[self._selected_q_idx, :]

    @q_values.setter
    def q_values(self, value: np.ndarray):
        self._q_values[self._selected_q_idx, :] = value

    @property
    def all_q_values(self) -> np.ndarray:
        """ Return the entire Q-values array without selecting the given index. """
        return self._q_values

    @property
    def action_visits(self) -> np.ndarray:
        """ Return number of time each action has been selected in this node for the given sampling. """
        return self._action_visits[self._selected_q_idx, :]

    @property
    def all_action_visits(self):
        """ Return all actions visits without selecting the appropriate sampling index. """
        return self._action_visits

    @property
    def q_index(self) -> int:
        """ The currently selected Q-value to update. """
        return self._selected_q_idx
