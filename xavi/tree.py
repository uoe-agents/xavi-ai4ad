from typing import Dict, Tuple, List, Union

import networkx as nx
import numpy as np
import igp2 as ip

from xavi.node import XAVINode
from xavi.util import Sample


class XAVITree(ip.Tree):
    """ Adds Q-value decomposition to the standard MCTS Tree for each possible trajectory sampling combination. """

    def __init__(self,
                 root: XAVINode,
                 action_policy: ip.Policy = None,
                 plan_policy: ip.Policy = None,
                 predictions: Dict[int, ip.GoalsProbabilities] = None):
        # Fields related to decomposing Q-values based on sampling combinations
        self._num_predictions = 1
        if len(predictions) > 0:
            # Assumes agent samplings are independent of one another
            #  Start with one extra to store overall Q-values.
            self._num_predictions = len(Sample.all_combinations(predictions))
            for aid, p in predictions.items():
                num_trajectories = sum([len(x) for x in p.all_trajectories.values()])
                self._num_predictions += len(p.goals_and_types) * np.product(num_trajectories)
        root.expand_samples(self._num_predictions)
        self._samples_map = []

        super(XAVITree, self).__init__(root, action_policy, plan_policy, predictions)

    def add_child(self, parent: XAVINode, child: XAVINode):
        """ Add a child node to an existing parent node. Extend Q-value array with number of possible samplings. """
        child.expand_samples(self._num_predictions)
        child.select_q_idx(parent.q_index)
        super(XAVITree, self).add_child(parent, child)

    def set_samples(self, samples: Union[Dict[int, Tuple[ip.GoalWithType, ip.VelocityTrajectory]], Sample]):
        """ Set the current sample in the Tree and updated the sample mapping.

        Args:
            samples: Either a dictionary of samples for each agent, or a Sample object.
                If None, then use overall Q-values.
        """
        if samples is None:
            self._samples = None
            self.root.select_q_idx(-1)
            return
        if isinstance(samples, dict):
            samples = Sample(samples)

        super(XAVITree, self).set_samples(samples.samples)

        if samples not in self._samples_map:
            self._samples_map.append(samples)
            s_idx = len(self._samples_map) - 1
        else:
            s_idx = self._samples_map.index(samples)

        assert s_idx != self._num_predictions - 1, "Last row of Q-values cannot be selected through samples. " \
                                                   "It is the overall running Q-value."
        self.root.select_q_idx(s_idx)

    def select_plan(self) -> List:
        """ Select optimal plan using overall Q-values. """
        self.root.select_q_idx(-1)
        return super(XAVITree, self).select_plan()

    def select_action(self, node: XAVINode) -> ip.MCTSAction:
        """ Select an action using overall Q-values while updating both overall and current-sampling Q-values. """
        current_q_idx = self.root.q_index
        self.root.select_q_idx(-1)

        action, idx = self._action_policy.select(node)
        node.action_visits[idx] += 1

        self.root.select_q_idx(current_q_idx)
        node.action_visits[idx] += 1

        return action

    def backprop(self, r: float, final_key: Tuple):
        """ Backpropagate both overall Q-values and current-sampling Q-values. """
        # First backprop on the current selected sample
        super(XAVITree, self).backprop(r, final_key)

        # Then backprop as normal
        current_q_idx = self.root.q_index
        self.root.select_q_idx(-1)
        super(XAVITree, self).backprop(r, final_key)
        self.root.select_q_idx(current_q_idx)

    @property
    def graph(self) -> nx.Graph:
        """ Returns a NetworkX graph representation of the search tree. """
        g = nx.DiGraph()
        for key, node in self.tree.items():
            g.add_node(key)
            for action in node.actions_names:
                child_key = key + (action, )
                g.add_node(child_key)
                g.add_edge(key, child_key, action=action)
        return g

    @property
    def root(self) -> XAVINode:
        """ The root node. """
        return self._root

    @property
    def samples_map(self) -> List[Sample]:
        """ Returns a list of all sampling combinations that were used at one point when running MCTS. """
        return self._samples_map
