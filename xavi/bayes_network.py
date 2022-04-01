import logging

from typing import List, Dict, Union, Optional

import igp2 as ip
import more_itertools
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

from xavi.tree import XAVITree
from xavi.util import Normal, Sample

logger = logging.getLogger(__name__)


class XAVIBayesNetwork:
    """ Create a Bayesian network model from a MCTS search tree. """

    def __init__(self,
                 alpha: float = 0.005,
                 use_log: bool = False,
                 reward_bins: int = 10,
                 cov_reg: float = 1e-3):
        """ Initialise a new Bayesian network from MCTS search results.

        Args:
            alpha: Add-alpha smoothing parameter.
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
        self._bn = None

        # Variables to store calculated probabilities
        self._p_t = {}
        self._p_omega_t = {}
        self._p_omega = {}
        self._p_r_omega = {}
        self._p_r = {}

        # To store discretised reward-related values
        self._reward_bin_params = {}
        self._reward_to_bin = {}

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
        self._discretize_rewards()

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
            for d in range(1, self._tree.max_depth + 1):
                for node in self._tree.nodes_at_depth(d):
                    for action in node.actions_names + [None]:
                        child_key = node.key + (action,)
                        p = self.p_omega_t(child_key, sample)
                        self._p_omega_t[sample][child_key] = p

            # Calculate unconditional probability as we are summing over samples
            for action, p in self._p_omega_t[sample].items():
                parent_p = 1.0 if not self.use_log else 0.0
                parent_key = action[:-1]
                while parent_key != self._tree.root.key:
                    pp = self._p_omega_t[sample][parent_key]
                    parent_p = parent_p * pp if not self.use_log else parent_p + np.log(pp)
                    parent_key = parent_key[:-1]

                if action not in self._p_omega:
                    self._p_omega[action] = 0.0
                self._p_omega[action] += p * p_sample * parent_p if not self.use_log else p + p_sample + parent_p

    def _calc_rewards(self):
        # Empty reward dictionary to get all possible rewards components
        comp_dict = ip.Reward().reward_components

        for actions, p_omega in self._p_omega.items():
            p = self.p_r_omega(actions, pdf=True, **comp_dict)
            self._p_r_omega[actions] = p

            # Calculate unconditional probability by summing over actions
            for comp, pdf in p.items():
                p_joint = pdf * p_omega  # TODO: Add log support
                if comp not in self._p_r:
                    self._p_r[comp] = p_joint
                else:
                    self._p_r[comp] += p_joint

    def _discretize_rewards(self):
        for comp in self._p_r:
            # Calculate bins for each reward component
            loc_scale_arr = [(x[comp].loc, x[comp].scale) for x in self._p_r_omega.values() if x[comp].loc is not None]
            if not loc_scale_arr:  # Arbitrary states for reward which has never been observed
                low, high = 0, 10
            else:
                low, low_scale = np.min(loc_scale_arr, axis=0)
                high, high_scale = np.max(loc_scale_arr, axis=0)
                low = low - 2 * high_scale
                high = high + 2 * high_scale
            self._reward_bin_params[comp] = (low, high)
            bins = np.arange(low, high, (high - low) / self.reward_bins)
            self._reward_to_bin[comp] = lambda x: np.digitize(x, bins)

    def p_t(self, agent_id: int, goal: ip.GoalWithType, trajectory: ip.VelocityTrajectory) -> float:
        """ Calculate goal-trajectory joint probability for a given agent.

         Args:
             agent_id: ID of the agent
             goal: Goal to condition on
             trajectory: A possible trajectory to the goal
         """
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
        independently of one another.

        Args:
            sample: A joint sample for some agents
        """
        p_joint = 1.0 if not self.use_log else 0.0
        for aid, (goal, trajectory) in sample.samples.items():
            p = self.p_t(aid, goal, trajectory)
            p_joint = p_joint * p if not self.use_log else p_joint + p
        return p_joint

    def p_omega_t(self, actions: List[str], sample: Sample) -> float:
        """ Calculate the conditional probability of a macro action given a preceding sequence of macro actions
        and given some sampling.

        Args:
            actions: A list of MA-keys from MCTS, where the final MA is conditioned on the preceding MAs.
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

        idx = node.actions_names.index(action) if action is not None else -1
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
        assert node is not None, f"Node key {key} not found in search tree."

        if action is None:
            if pdf:
                return {comp: Normal(None, None) for comp in rewards}
            return {comp: Normal(None, None).pdf(val) for comp, val in rewards.items()}

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

    def p_o_r(self, outcome: str = None, **rewards) -> Union[float, Dict[str, float]]:
        """ DO NOT USE. USES WRONG METHOD TO CALCULATE OUTCOME PROBABILITY.
        Calculate probability of an outcome given the rewards received.

        Args:
            outcome: The type of the outcome. Currently can be 'dead', 'term', 'coll', and 'done'.
            rewards: The values for each reward component of interest.
        """
        if outcome is not None and outcome not in self._outcome_reward_map:
            logger.debug(f"Invalid outcome type {outcome} given.")
            return 0.0 if not self.use_log else -np.inf

        p_comps = {}
        for oc, comps in self._outcome_reward_map.items():
            val = None
            for comp in comps:
                if comp not in rewards:
                    continue
                p = self._p_r[comp].pdf(rewards[comp])
                if not self.use_log:
                    val = p if val is None else val * p
                else:
                    val = np.log(p) if val is None else val + np.log(p)
            p_comps[oc] = val if val is not None else (0.0 if not self.use_log else -np.inf)

        norm_factor = sum(p_comps.values())
        if np.isnan(norm_factor):
            return 0.0
        if outcome is None:
            return {k: v / norm_factor for k, v in p_comps.items()}
        return p_comps[outcome] / norm_factor

    def p_o(self):
        """ DO NOT USE. USES WRONG METHOD FOR OUTCOME PROBABILITY.
        Unconditional probabilities of outcomes. """
        ret = {}
        p_r = np.prod([v for v in self._p_r.values()])
        for outcome, comps in self._outcome_reward_map.items():
            p_o = np.prod([self._p_r[comp] for comp in comps]) * p_r
            ret[outcome] = p_o.integrate()

        norm_factor = sum(ret.values())
        return {k: v / norm_factor for k, v in ret.items()}

    def reward_to_bin(self, x: float, reward: str) -> int:
        """ Return which discretised reward bin the given value belongs into.

        Args:
            x: Value to bin
            reward: The reward type
        """
        return self._reward_to_bin[reward](x)

    def to_bayesian_network(self) -> BayesianNetwork:
        """ Generate all conditional tables for the Bayesian network and
        create an explicit pgmpy.BayesianNetwork object. Stored in self.bn.

        Returns:
            A pgmpy.BayesianNetwork object
        """
        def add_cpd(var, ev):
            bn.add_cpds(TabularCPD(variable=var,
                                   variable_card=cardinalities[var],
                                   values=values[var].reshape(cardinalities[var], -1),
                                   evidence=ev,
                                   evidence_card=[cardinalities[k] for k in ev],
                                   state_names={k: states[k] for k in [var] + ev}))
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
                                   state_names={gn: states[gn]}))

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
            add_cpd(tn, [gn])

        # Add nodes and edges for macro actions
        condition_set = [f"trajectory_{aid}" for aid in self._p_t]
        for d in range(1, (self._tree.max_depth + 1)):
            on = f"omega_{d}"
            bn.add_edges_from([(cond, on) for cond in condition_set])
            possible_mas = self._tree.actions_at_depth(d) + [None]
            cardinalities[on] = len(possible_mas)
            states[on] = possible_mas
            values[on] = np.zeros([cardinalities[cond] for cond in condition_set] + [cardinalities[on]])
            values[on][..., -1] = 1.0  # Pre-set the empty action to have probability one. Will be overridden later.

            for sample, actions in self._p_omega_t.items():
                samples_idx = []
                for aid, (goal, trajectory) in sample.samples.items():
                    samples_idx.append(states[f"trajectory_{aid}"].index(trajectory))
                samples_idx = tuple(samples_idx)
                for action, p in actions.items():
                    if len(action) - 1 != d:  # Subtract one for Root key
                        continue
                    action_idx = tuple([states[f"omega_{dd}"].index(ma) for dd, ma in enumerate(action[1:], 1)])
                    cpd = values[on]
                    cpd[samples_idx + action_idx] = p

            values[on] = values[on].reshape(-1, cardinalities[on]).T
            add_cpd(on, condition_set)
            condition_set.append(on)

        # Add nodes for reward components
        condition_set = [k for k in values if k.startswith("omega")]
        for comp in self._p_r:
            rn = f"reward_{comp}"
            bn.add_edges_from([(cond, rn) for cond in condition_set])
            cardinalities[rn] = self.reward_bins + 1  # Add one for no rewards
            cpd = np.zeros([cardinalities[cond] for cond in condition_set] + [cardinalities[rn]])
            cpd[..., -1] = 1.0
            values[rn] = cpd
            low, high = self._reward_bin_params[comp]
            states[rn] = list(np.arange(low, high, (high - low) / self.reward_bins)) + [None]

        for action, pdfs in self._p_r_omega.items():
            action_idx = [states[f"omega_{d}"].index(ma) for d, ma in enumerate(action[1:], 1)]
            action_idx += [-1] * (len(condition_set) - len(action_idx))  # Pad the rest with the empty action
            action_idx = tuple(action_idx)

            for comp, pdf in pdfs.items():
                rn = f"reward_{comp}"
                low, high = self._reward_bin_params[comp]
                if pdf.loc is not None:
                    discrete_values, _ = pdf.discretize(low=low, high=high, bins=self.reward_bins, norm=True)
                    values[rn][action_idx][:-1] = discrete_values
                    values[rn][action_idx][-1] = 0.0

        for comp in self._p_r:
            rn = f"reward_{comp}"
            values[rn] = values[rn].reshape(-1, cardinalities[rn]).T
            add_cpd(rn, condition_set)

            # Add dummy variables to binarise whether a reward is present or not
            brn = f"b{rn}"
            cardinalities[brn] = 2
            states[brn] = [False, True]
            values[brn] = np.zeros((cardinalities[brn], cardinalities[rn]))
            values[brn][0, -1] = 1.0
            values[brn][1, :-1] = 1.0
            bn.add_edge(rn, brn)
            add_cpd(brn, [rn])

        # Add the outcome variable
        #  Outcome variables currently assume that associated sets of reward components are mutually exclusive.
        for outcome, rewards in self._outcome_reward_map.items():
            condition_set = [f"breward_{cond}" for cond in rewards]
            on = f"outcome_{outcome}"
            cardinalities[on] = 2  # Binary variable
            states[on] = [False, True]
            values[on] = np.zeros([cardinalities[cond] for cond in [on] + condition_set])
            values[on][0, (0, ) * len(condition_set)] = 1.0  # Outcome false iff all associated rewards are False
            values[on][1, ...] = 1.0  # Otherwise it is true
            values[on][1, (0, ) * len(condition_set)] = 0.0
            bn.add_edges_from([(cond, on) for cond in condition_set])
            add_cpd(on, condition_set)

        self._bn = bn
        return self._bn

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
    def bn(self) -> Optional[BayesianNetwork]:
        """ The explicit BN-representation of this network. If to_bayesian_network() has not been called,
         then will return None. """
        return self._bn

    @property
    def outcome_to_reward(self) -> Dict[str, List[str]]:
        """ Defines a mapping from outcome types to reward types. """
        return self._outcome_reward_map
