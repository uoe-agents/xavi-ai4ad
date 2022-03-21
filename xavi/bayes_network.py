import logging

from typing import List, Dict, Union

import igp2 as ip
import more_itertools
import numpy as np
from bidict import bidict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.continuous import CanonicalDistribution, ContinuousFactor, RoundingDiscretizer
from pgmpy.factors.distributions import GaussianDistribution
from pgmpy.models import BayesianNetwork

from xavi.tree import XAVITree
from xavi.util import Normal, Sample

logger = logging.getLogger(__name__)


class XAVIBayesNetwork:
    """ Create a Bayesian network model from a MCTS search tree. """

    def __init__(self,
                 alpha: float = 2.0,
                 use_log: bool = False,
                 reward_bins: int = 10,
                 cov_reg: float = 1e-3):
        """ Initialise a new Bayesian network from MCTS search results.

        Args:
            alpha: Scaling parameter to use in softmax.
            use_log: Whether to work in log-space.
            reward_bins: If not None, then discretise the normal distributions of rewards into this many bins.
            cov_reg: The covariance regularisation factor for reward components.
        """
        self.alpha = alpha
        self.use_log = use_log
        self.reward_bins = reward_bins
        self.cov_reg = cov_reg

        self._outcome_reward_map = {
            "dead": ["dead"],
            "coll": ["coll"],
            "term": ["term"],
            "done": ["time", "jerk", "angular_acceleration", "curvature"]
        }

        self._results = None
        self._tree = None

        # Variables to store calculated probabilities
        self._p_t = {}
        self._p_omega_t = {}
        self._p_omega = {}
        self._p_r_omega = {}
        self._p_r = {}

    def update(self, mcts_results: ip.AllMCTSResult):
        """ Overwrite the currently stored MCTS results and calculate the BN probabilities from it.

        Args:
            mcts_results: Class containing all relevant results of the MCTS run.
        """
        self._results = mcts_results
        self._tree = mcts_results[-1].tree

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
            p_sample = self.p_t_joint(sample)
            self._p_omega_t[sample] = {}
            for key, node in self._tree.tree.items():
                for action in node.actions_names:
                    child_key = key + (action,)
                    p = self.p_omega_t(child_key, sample)
                    self._p_omega_t[sample][child_key] = p

                    # Calculate unconditional probability by summing over samples
                    if child_key not in self._p_omega:
                        self._p_omega[child_key] = 0.0
                    self._p_omega[child_key] += p * p_sample if not self.use_log else p + p_sample

    def _calc_rewards(self):
        # Empty reward dictionary to get all possible rewards components
        comp_dict = ip.Reward().reward_components

        for actions, p_omega in self._p_omega.items():
            p = self.p_r_omega(actions, pdf=True, **comp_dict)
            self._p_r_omega[actions] = p

            # Calculate unconditional probability by summing over actions
            for comp, pdf in p.items():
                p_joint = pdf * p_omega
                if comp not in self._p_r:
                    self._p_r[comp] = p_joint
                else:
                    self._p_r[comp] += p_joint

    def p_t(self, agent_id: int, goal: ip.GoalWithType, trajectory: ip.VelocityTrajectory) -> float:
        """ Calculate goal-trajectory joint probability for a given agent """
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

    def p_t_joint(self, sample: Sample) -> float:
        """ Return the joint probability of the given Sample assuming that agents are sampled
        independently of one another. """
        p_joint = 1.0 if not self.use_log else 0.0
        for aid, (goal, trajectory) in sample.samples.items():
            p = self.p_t(aid, goal, trajectory)
            p_joint = p_joint * p if not self.use_log else p_joint + p
        return p_joint

    def p_omega_t(self, actions: List[str], sample: Sample) -> float:
        """ Calculate the conditional probability of a sequence of macro actions given some sampling.

        Args:
            actions: A list of MA-keys from MCTS.
            sample: The sampling to condition on.
        """
        if actions[0] != self._tree.root.key[0]:
            actions.insert(0, self._tree.root.key[0])
        key = tuple(actions)

        if sample in self._p_omega_t and key in self._p_omega_t[sample]:
            return self._p_omega_t[sample][key]

        self._tree.set_samples(sample)

        node, action, child = (self._tree[key[:-1]], key[-1], self._tree[key])
        if node is None:
            logger.debug(f"Node key {key} not found. Returning zero probability.")
            return 0.0 if not self.use_log else -np.inf

        idx = node.actions_names.index(action)
        action_prob = node.action_probabilities(self.alpha)[idx]
        prob = action_prob if not self.use_log else np.log(action_prob)
        return prob

    def p_omega(self, actions: List[str] = None):
        """ Calculate the unconditional probability of the given sequence of macro actions
         by summing over all sampling combinations.

         Args:
             actions: The sequence of macro actions. If None, return the whole distribution.
         """
        if actions is None:
            return self._p_omega

        key = tuple(actions)
        p = self._p_omega.get(key, None)
        if p is None:
            logger.debug(f"Actions {key} not a valid sequence of actions in the tree.")
            return 0.0 if not self.use_log else -np.inf
        return p

    # TODO: Possibly consider adding cost function elements not only reward function elements,
    #  as rewards are harder to interpret. Otherwise, add method to convert cost elements to rewards automatically.
    def p_r_omega(self, actions: List[str], pdf: bool = False, **rewards) \
            -> Dict[str, Union[Normal, float]]:
        """ Calculate probability of receiving rewards given the actions. The reward components can be specified as
        keyword argments.

        Keyword Args:
            coll: Ego collision
            term: Termination by reaching search depth
            dead: Ego died due something other than a collision
            time: Time to goal
            jerk: Average jerk
            angular_acceleration: Average angular acceleration
            curvature: Average trajectory curvature

        Args:
            actions: List of MA-keys from MCTS
            pdf: If True, return the PDFs instead of the likelihood
            rewards: A dictionary of reward components.

        Returns:
            A dictionary of reward likelihoods/PDFs
        """
        if actions[0] != self._tree.root.key[0]:
            actions.insert(0, self._tree.root.key[0])
        key = tuple(actions)
        action = actions[-1]
        node = self._tree[key[:-1]]

        if node is None:
            logger.debug(f"Node key {key} not found in search tree.")
            return 0.0 if not self.use_log else -np.inf

        if key in self._p_r_omega:
            if pdf:
                return {comp: dist for comp, dist in self._p_r_omega[key].items() if comp in rewards}
            return {comp: dist.pdf(rewards[comp]) for comp, dist in self._p_r_omega[key].items() if comp in rewards}

        # Accumulate all rewards for each component
        reward_results = node.reward_results[action]
        r_dists = {comp: [] for comp in rewards}
        for reward in reward_results:
            for component, value in reward.reward_components.items():
                r_dists[component].append(value)

        # Calculate mean and variance in reward component
        ret = {}
        for component, value in r_dists.items():
            filtered = [v for v in value if v is not None]
            if len(filtered) == 0:
                mean, std = None, None
            else:
                mean, std = float(np.mean(filtered)), np.std(filtered) + self.cov_reg
            ret[component] = Normal(mean, std)

        if pdf:
            return ret
        return {comp: dist.pdf(rewards[comp]) for comp, dist in ret.items()}

    def p_r(self, pdf: bool = False, **rewards) -> Dict[str, Union[Normal, float]]:
        """ Return the unconditional probability distribution for the given rewards.
        If no keyword arguments are passed then return every PDF for each reward component.

        Args:
            pdf: If True then return the PDFs only without evaluation.

        Keyword Args:
            coll: Ego collision
            term: Termination by reaching search depth
            dead: Ego died due something other than a collision
            time: Time to goal
            jerk: Average jerk
            angular_acceleration: Average angular acceleration
            curvature: Average trajectory curvature
        """
        if rewards is None:
            if not pdf:
                logger.warning(f"Argument pdf was True, but no reward components were passed.")
            return self._p_r

        if pdf:
            return {comp: self._p_r[comp] for comp in rewards}
        return {comp: self._p_r[comp].pdf(val) for comp, val in rewards.items()}

    def p_o_r(self, outcome: str, **rewards) -> Union[float, Dict[str, float]]:
        """ Calculate probability of an outcome given the rewards received.

        Args:
            outcome: The type of the outcome. Currently can be 'dead', 'term', 'coll', and 'done'.
            rewards: The values for each reward component of interest.
        """
        if outcome not in self._outcome_reward_map:
            logger.debug(f"Invalid outcome type {outcome} given.")
            return 0.0 if not self.use_log else -np.inf

        p_comps = {}
        for oc, comps in self._outcome_reward_map.items():
            val = None
            for comp in comps:
                if comp not in rewards:
                    logger.warning(f"Reward component {comp} for outcome was not found "
                                   f"in the passed rewards dictionary.")
                    continue
                p = self._p_r[comp].pdf(rewards[comp])
                if not self.use_log:
                    val = p if val is None else val * p
                else:
                    val = np.log(p) if val is None else val + np.log(p)
            p_comps[oc] = val if val is not None else (0.0 if not self.use_log else -np.inf)

        norm_factor = sum(p_comps.values())
        if outcome is None:
            return {k: v / norm_factor for k, v in p_comps.items()}
        return p_comps[outcome] / norm_factor

    def p_o(self):
        """ Unconditional probabilities of outcomes. """
        ret = {}
        p_r = np.prod([v for v in self._p_r.values()])
        for outcome, comps in self._outcome_reward_map.items():
            p_o = np.prod([self._p_r[comp] for comp in comps]) * p_r
            ret[outcome] = p_o.integrate()

        norm_factor = sum(ret.values())
        return {k: v / norm_factor for k, v in ret.items()}

    def to_bayesian_network(self) -> BayesianNetwork:
        """ Generate all conditional tables for the Bayesian network and
        create an explicit pgmpy.BayesianNetwork object. """
        bn = BayesianNetwork()
        cardinalities = {}
        states = {}
        values = {}

        # Add sampling nodes and conditional tables
        for aid, goals in self._p_t.items():
            gn, tn = f"goal_{aid}", f"trajectory_{aid}"
            bn.add_edge(gn, tn)
            # Add goal priors
            states[gn] = list(goals)
            cardinalities[gn] = len(states[gn])
            values[gn] = np.array([[sum(t.values())] for t in goals.values()])
            bn.add_cpds(TabularCPD(variable=gn,
                                   variable_card=cardinalities[gn],
                                   values=values[gn],
                                   state_names={gn: list(map(str, states[gn]))}))

            # Add conditional trajectory probabilities
            states[tn] = list(more_itertools.flatten(goals.values())) + [None]
            cardinalities[tn] = len(states[tn])
            cpd = np.zeros((cardinalities[tn], cardinalities[gn]))
            cpd[-1, ...] = 1.0
            for i, g in enumerate(states[gn]):
                for j, t in enumerate(states[tn]):
                    try:
                        p = goals[g][t]
                        cpd[j, i] = p / values[gn][i, 0]  # Divide by goal prior since p is a joint
                        cpd[-1, i] = 0.0
                    except KeyError:
                        continue
            values[tn] = cpd
            bn.add_cpds(TabularCPD(variable=tn,
                                   variable_card=cardinalities[tn],
                                   values=cpd,
                                   evidence=[gn],
                                   evidence_card=[cardinalities[gn]],
                                   state_names={gn: list(map(str, states[gn])),
                                                tn: list(map(str, states[tn]))}))

        # Add nodes and edges for macro actions
        condition_set = [f"trajectory_{aid}" for aid in self._p_t]
        possible_mas = self.macro_actions + [None]
        for d in range(1, (self._tree.max_depth + 1) + 1):
            on = f"omega_{d}"
            bn.add_edges_from([(cond, on) for cond in condition_set])
            cardinalities[on] = len(possible_mas)
            states[on] = possible_mas
            values[on] = np.zeros([cardinalities[cond] for cond in condition_set] + [cardinalities[on]])
            values[on][..., -1] = 1.0  # Pre-set the empty action to have probability one. Will be overridden later.
            condition_set.append(on)

        for sample, actions in self._p_omega_t.items():
            samples_idx = []
            for aid, (goal, trajectory) in sample.samples.items():
                samples_idx.append(states[f"trajectory_{aid}"].index(trajectory))
            samples_idx = tuple(samples_idx)

            for action, p in actions.items():
                d = len(action) - 1  # Subtract one for 'Root' node
                on = f"omega_{d}"
                action_idx = tuple([possible_mas.index(ma) for ma in action[1:]])

                cpd = values[on]
                cpd[samples_idx + action_idx] = p
                cpd[samples_idx + action_idx[:-1] + (-1,)] = 0.0  # Empty action has probability zero

        condition_set = [f"trajectory_{aid}" for aid in self._p_t]
        for d in range(1, (self._tree.max_depth + 1) + 1):
            on = f"omega_{d}"
            value = values[on].reshape(-1, cardinalities[on]).T
            bn.add_cpds(TabularCPD(variable=on,
                                   variable_card=cardinalities[on],
                                   values=value,
                                   evidence=condition_set,
                                   evidence_card=[cardinalities[cond] for cond in condition_set],
                                   state_names={cond: list(map(str, states[cond])) for cond in condition_set + [on]}))
            condition_set.append(on)

        # Add nodes for reward components
        value_to_bin = {}
        condition_set = [k for k in values if k.startswith("omega")]
        for comp in self._p_r:
            rn = f"reward_{comp}"
            bn.add_edges_from([(cond, rn) for cond in condition_set])
            cardinalities[rn] = self.reward_bins + 1  # Add one for no rewards
            cpd = np.zeros([cardinalities[cond] for cond in condition_set] + [cardinalities[rn]])
            cpd[..., -1] = 1.0
            values[rn] = cpd
            value_to_bin[comp] = {}

        for actions, pdfs in self._p_r_omega.items():
            action_idx = [possible_mas.index(ma) for ma in actions[1:]]
            action_idx += [-1] * (len(condition_set) - len(action_idx))  # Pad the rest with the empty action
            action_idx = tuple(action_idx)

            for comp, pdf in pdfs.items():
                rn = f"reward_{comp}"
                loc, scale = pdf.loc, pdf.scale
                if loc is not None and scale is not None:
                    discrete_values, bins = pdf.discretize(low=loc - self.reward_bins / 2 * scale,
                                                           high=loc + self.reward_bins / 2 * scale,
                                                           bins=self.reward_bins)
                    values[rn][action_idx][:-1] = discrete_values
                    values[rn][action_idx][-1] = 0.0
                    value_to_bin[comp][actions] = bins
                else:
                    bins = np.arange(0, 1, 1 / 10)  # Use some arbitrary label for None reward component
                states[rn] = list(bins) + [None]

        for comp in self._p_r:
            rn = f"reward_{comp}"
            value = values[rn].reshape(-1, cardinalities[rn]).T
            bn.add_cpds(TabularCPD(variable=rn,
                                   variable_card=cardinalities[rn],
                                   values=value,
                                   evidence=condition_set,
                                   evidence_card=[cardinalities[cond] for cond in condition_set],
                                   state_names={cond: list(map(str, states[cond])) for cond in condition_set + [rn]}))


    @property
    def tree(self) -> XAVITree:
        """ The MCTS search tree associated with this network. """
        return self._tree

    @property
    def macro_actions(self) -> List[str]:
        """ A list of all macro actions occurring in the MCTS tree. """
        mas = list(set(more_itertools.flatten(self._p_omega)))
        mas.remove("Root")
        return mas

    @property
    def outcome_to_reward(self) -> Dict[str, List[str]]:
        """ Defines a mapping from outcome types to reward types. """
        return self._outcome_reward_map
