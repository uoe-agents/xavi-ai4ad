from typing import Tuple, Dict, List

import igp2 as ip
import numpy as np

from xavi.util import softmax


class XAVINode(ip.Node):
    """ Subclass of ip.Node that allows decomposition of Q-values based on non-ego samples. """

    def __init__(self, key: Tuple, state: Dict[int, ip.AgentState], actions: List[ip.MCTSAction]):
        super(XAVINode, self).__init__(key, state, actions)
        self._selected_q_idx = None
        self._dnf = None

    def expand_samples(self, num_samples: int):
        if self._actions is None:
            raise TypeError("Cannot expand node without actions")
        self._q_values = np.zeros((num_samples, len(self._actions)))
        self._action_visits = np.zeros((num_samples, len(self._actions)), dtype=np.int32)
        self._dnf = np.zeros(num_samples)

    def select_q_idx(self, idx: int = 0):
        """ Specify which Q-values to update, and update all children recurisvely nodes as well. """
        self._selected_q_idx = idx
        for child in self.children.values():
            child.select_q_idx(idx)

    def action_probabilities(self, alpha: float = 1.0):
        """ Get the current action probabilities for the selected sampling index based on
        relative frequency of action visits.

        Args:
            alpha: Smoothing parameter in the softmax.
        """
        dnf = self._dnf
        visits = self._action_visits
        if len(self._key) == 1:  # Empty action is not allowed at the root node
            probs = (visits + alpha) / np.sum(visits + alpha, axis=1, keepdims=True)
            probs = np.hstack([probs, np.zeros_like(dnf)[:, None]])
        else:
            possible_actions = visits.sum(1) > 0
            dnf[~possible_actions] = 1  # Set no-action to be most probable in samples that were never seen
            visits = np.hstack([visits, dnf[:, None]])
            probs = (visits + alpha) / np.sum(visits + alpha, axis=1, keepdims=True)

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

    @property
    def dnf(self) -> np.ndarray:
        """ Number of times the simulation terminated at this node without the run reaching maximum tree depth. """
        return self._dnf
