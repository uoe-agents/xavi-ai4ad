import logging

from typing import List, Dict
from scipy.stats import norm

import igp2 as ip
import numpy as np

from xavi.tree import XAVITree
from xavi.util import Sample

logger = logging.getLogger(__name__)


class XAVIBayesNetwork:
    """ Create a Bayesian network model from a MCTS search tree. """

    def __init__(self,
                 alpha: float = 5.0,
                 use_log: bool = False,
                 pre_calculate: bool = True):
        """ Initialise a new Bayesian network from MCTS search results.

        Args:
            alpha: Scaling parameter to use in softmax.
            use_log: Whether to work in log-space.
            pre_calculate: Calculate probabilities for all random variables in advance.
        """
        self.alpha = alpha
        self.use_log = use_log

        self._results = None
        self._tree = None

        # Variables to store calculated probabilities
        self._p_t = {}
        self._p_omega = {}
        self._p_r = {}

    def update(self, mcts_results: ip.AllMCTSResult, pre_calculate: bool = True):
        """ Overwrite the currently stored MCTS results and calculate the BN probabilities from it.

        Args:
            mcts_results: Class containing all relevant results of the MCTS run.
            pre_calculate: If True calculate probabilities
        """
        self._results = mcts_results
        self._tree = mcts_results[-1].tree
        if pre_calculate:
            self._calc_sampling()
            self._calc_actions()
            self._calc_rewards()

    def _calc_sampling(self):
        for aid, pred in self._tree.predictions.items():
            self._p_t[aid] = {}
            for goal, trajectories in pred.all_trajectories.items():
                self._p_t[aid][goal] = {}
                for trajectory in trajectories:
                    self._p_t[aid][goal][trajectory] = self.p_t(aid, goal, trajectory)

    def _calc_actions(self):
        for sample in self._tree.possible_samples:
            self._p_omega[sample] = {}
            for key, node in self._tree.tree.items():
                for action in node.actions_names:
                    child_key = key + (action, )
                    self._p_omega[sample][child_key] = self.p_omega(child_key, sample)

    def _calc_rewards(self):
        for key, node in self._tree.tree.items():
            for action in node.actions_names:
                child_key = key + (action, )
                self._p_r[child_key] = node.reward_results

    def p_t(self, agent_id: int, goal: ip.GoalWithType, trajectory: ip.VelocityTrajectory) -> float:
        """ Calculate all goal-trajectory joint probabilities for a given agent """
        if agent_id in self._p_t and \
                goal in self._p_t[agent_id] and \
                trajectory in self._p_t[agent_id][goal]:
            return self._p_t[agent_id][goal][trajectory]

        trajectory_idx = self._tree.predictions[agent_id].all_trajectories[goal].index(trajectory)
        assert trajectory_idx > -1, "Invalid trajectory given."

        p_goal = self._tree.predictions[agent_id].goals_probabilities[goal]
        p_trajectory = self._tree.predictions[agent_id].trajectories_probabilities[goal][trajectory_idx]
        return p_goal * p_trajectory if not self.use_log \
            else np.log(p_goal) + np.log(p_trajectory)

    def p_omega(self, actions: List[str], sample: Sample) -> float:
        """ Calculate the conditional probability of a sequence of macro actions given some sampling.

        Args:
            actions: A list of MA-keys from MCTS.
            sample: The sampling to condition on.
        """
        if actions[0] != self._tree.root.key[0]:
            actions.insert(0, self._tree.root.key[0])

        if sample in self._p_omega and actions in self._p_omega[sample]:
            return self._p_omega[sample][actions]

        self._tree.set_samples(sample)

        prob = 0.0 if self.use_log else 1.0
        key = tuple(actions)
        while key != self._tree.root.key:
            node, action, child = (self._tree[key[:-1]], key[-1], self._tree[key])
            if node is None:
                logger.debug(f"Node key {key} not found. Returning zero probability.")
                return 0.0 if not self.use_log else -np.inf

            idx = node.actions_names.index(action)
            action_prob = node.action_probabilities(self.alpha)[idx]
            if self.use_log:
                prob += np.log(action_prob)
            else:
                prob *= action_prob

            key = node.key

        return prob

    def p_r(self, actions: List[str], **rewards: Dict[str, float]) -> float:
        """ Calculate probability of receiving rewards given the actions. The reward components can be specified as
        keyword argments.

        Keyword Args:
            coll: Ego collision
            term: Termination by reaching search depth
            dead: Ego died due something other than a collision
            cost: Overall cost. This may be replaced by decomposing into the elements shown below.
            time: Time to goal
            velocity: Average velocity
            acceleration: Average acceleration
            jerk: Average jerk
            heading: Average heading
            angular_velocity: Average angular velocity
            angular_acceleration: Average angular acceleration
            curvature: Average trajectory curvature

        Args:
            actions: List of MA-keys from MCTS
            rewards: A dictionary of reward components.
        """
        if actions[0] != self._tree.root.key[0]:
            actions.insert(0, self._tree.root.key[0])
        key = tuple(actions)
        node = self._tree[key]

        if node is None:
            logger.debug(f"Node key {key} not found in search tree.")
            return 0.0 if not self.use_log else -np.inf



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
