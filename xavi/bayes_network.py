from typing import Union, List

import igp2 as ip
import numpy as np

from xavi.tree import XAVITree
from xavi.util import Sample


class XAVIBayesNetwork:
    """ Create a Bayesian network model from a MCTS search tree. """

    def __init__(self, alpha: float = 5.0, use_log: bool = False):
        """ Initialise a new Bayesian network from MCTS search results.

        Args:
            alpha: Scaling parameter to use in softmax.
            use_log: Whether to work in log-space.
        """
        self.alpha = alpha
        self.use_log = use_log

        self._results = None
        self._tree = None

        # Variables to store calculated probabilities
        self._p_t = {}

    def update(self, mcts_results: ip.AllMCTSResult, pre_calculate: bool = True):
        """ Overwrite the currently stored MCTS results and calculate the BN probabilities from it.

        Args:
            mcts_results: Class containing all relevant results of the MCTS run.
            pre_calculate: If True calculate probabilities
        """
        self._results = mcts_results
        self._tree = mcts_results[-1].tree
        if pre_calculate:
            self.calculate_probabilities()

    def calculate_probabilities(self):
        """ Using the MCTS run results derive the probability distributions of BN. """
        self._calc_sampling()

    def _calc_sampling(self):
        self._p_sampling = {}  # Bold, capital T in paper
        for aid, pred in self._tree.predictions.items():
            self._p_sampling[aid] = {}
            for goal, trajectories in pred.all_trajectories.items():
                p_goal = pred.goals_probabilities[goal]
                p_trajectories = np.array(pred.trajectories_probabilities[goal])
                self._p_sampling[aid][goal] = p_goal * p_trajectories

    def p_t(self, agent_id: int, goal: ip.GoalWithType, trajectory: ip.VelocityTrajectory) -> float:
        """ Calculate all goal-trajectory joint probabilities for a given agent """
        trajectory_idx = self._tree.predictions[agent_id].all_trajectories[goal].index(trajectory)
        assert trajectory_idx > -1, "Invalid trajectory given."

        p_goal = self._tree.predictions[agent_id].goals_probabilities[goal]
        p_trajectory = self._tree.predictions[agent_id].trajectories_probabilities[goal]
        return p_goal * p_trajectory if not self.use_log \
            else np.log(p_goal) + np.log(p_trajectory)

    def p_omega(self, actions: List[str], sample: Sample) -> float:
        """ Calculate the conditional probability of a sequence of macro actions given some sampling.

        Args:
            actions: A list of MA-keys from MCTS.
            sample: The sampling to condition on.
        """
        self.tree.set_samples(sample)

        if actions[0] != self.tree.root.key:
            actions.insert(self.tree.root.key)

        prob = 0.0 if self.use_log else 1.0
        key = tuple(actions)
        while key != self.tree.root.key:
            node, action, child = (self[key[:-1]], key[-1], self[key])

            idx = node.actions_names.index(action)
            action_prob = node.action_probabilities(self.alpha)[idx]
            if self.use_log:
                prob += np.log(action_prob)
            else:
                prob *= action_prob

            key = node.key

        return prob

    def p_r(self, rewards: np.ndarray, actions: List[str]) -> float:
        """ Calculate probability of receiving rewards given the actions. """
        pass

    def p_o(self, outcome: str, rewards: np.ndarray) -> float:
        """ Calculate probability of an outcome given the rewards received. """
        pass

    def to_tabular(self):
        """ Convert all conditional distributions to tabular form. """
        pass

    @property
    def tree(self) -> XAVITree:
        """ The MCTS search tree associated with this network. """
        return self._tree
