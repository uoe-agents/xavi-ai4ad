from typing import Dict, Tuple, List, Union

import more_itertools
import networkx as nx
import igp2 as ip
import logging

from xavi.node import XAVINode
from xavi.util import Sample

logger = logging.getLogger(__name__)


class XAVITree(ip.Tree):
    """ Adds Q-value decomposition to the standard MCTS Tree for each possible trajectory sampling combination. """

    def __init__(self,
                 root: XAVINode,
                 action_policy: ip.Policy = None,
                 plan_policy: ip.Policy = None,
                 predictions: Dict[int, ip.GoalsProbabilities] = None):
        # Assumes agent samplings are independent of one another
        #  Start with one extra to store overall Q-values.
        self._num_predictions = 1
        if len(predictions) > 0:
            self._possible_samples = Sample.all_combinations(predictions)
            self._num_predictions += len(self._possible_samples)
        root.expand_samples(self._num_predictions)

        self._backprop_traces = []

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

        if samples not in self._possible_samples:
            raise RuntimeError(f"Sample not found in possible samples")
        else:
            s_idx = self._possible_samples.index(samples)

        assert s_idx != self._num_predictions - 1, "Last row of Q-values cannot be selected through samples. " \
                                                   "It is the overall running Q-value."

        logger.debug(f"Samples {s_idx} selected: {samples.samples}")
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
        """ Back-propagate both overall Q-values and current-sampling Q-values. """
        # First backprop on the current selected sample
        super(XAVITree, self).backprop(r, final_key)

        # Then backprop as normal
        current_q_idx = self.root.q_index
        self.root.select_q_idx(-1)
        super(XAVITree, self).backprop(r, final_key)
        self.root.select_q_idx(current_q_idx)

        self._backprop_traces.append((current_q_idx, final_key))

    def nodes_at_depth(self, d: int) -> List[XAVINode]:
        """ Return a list of nodes at the given depth, where the root has depth 1. """
        return [n for k, n in self._tree.items() if len(k) == d]

    def actions_at_depth(self, d: int) -> List[str]:
        """ Return a list of actions that are valid in some node at depth d, with the root having depth 1. """
        return list(set([an for n in self.nodes_at_depth(d) for an in n.actions_names]))

    def collision_from_node(self, key: Tuple) -> List[ip.Agent]:
        """ Return a list of colliding agents at the given node, the node's parent,
        or any of the node's descendants.

        Args:
            key: Key of the node to check collisions for
        """
        def add_collided_agents(agents, collided_agent_ids):
            for cid in collided_agent_ids:
                if cid not in cids:
                    ret.append(agents[cid])
                    cids.append(cid)

        node = self[key]
        assert node is not None, f"Node {node} not found in tree. "
        parent = self[key[:-1]]

        ret = []
        cids = []
        # Add collisions from parent where given action was chosen
        if parent is not None:
            for r in parent.run_results:
                if str(r.selected_action) != key[-1]:
                    continue
                add_collided_agents(r.agents, r.collided_agents_ids)

        for n in [node] + node.descendants:
            for r in n.run_results:
                add_collided_agents(r.agents, r.collided_agents_ids)

        return ret

    def on_finish(self):
        """ In this implementation, we look at all traces to see which one did not reach max search depth
        and accumulate that into the field 'dnf'. """
        for q_idx, trace in self._backprop_traces:
            if len(trace) - 1 < self.max_depth:
                self[trace[:-1]].dnf[q_idx] += 1
                self[trace[:-1]].dnf[-1] += 1

    @property
    def agents(self) -> Dict[int, ip.Agent]:
        """ Get all agents appearing during the search. """
        ret = {}
        for key, node in self._tree.items():
            for rr in node.run_results:
                for aid, agent in rr.agents.items():
                    if aid not in ret:
                        ret[aid] = agent
        return ret

    @property
    def graph(self) -> nx.Graph:
        """ Returns a NetworkX graph representation of the search tree. """
        g = nx.DiGraph()
        for key, node in self._tree.items():
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
    def backprop_traces(self) -> List[Tuple[int, str]]:
        """ The traces received during back-propagation. """
        return self._backprop_traces

    @property
    def possible_samples(self) -> List[Sample]:
        """ Returns a list of all possible sampling combinations that were used at one point when running MCTS. """
        return self._possible_samples
