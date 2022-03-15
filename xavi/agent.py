from typing import Dict

import igp2 as ip

from xavi.bayes_network import XAVIBayesNetwork
from xavi.node import XAVINode
from xavi.tree import XAVITree


class XAVIAgent(ip.MCTSAgent):
    """ Agent that gives explanations of its actions. """

    def __init__(self,
                 agent_id: int,
                 initial_state: ip.AgentState,
                 t_update: float,
                 scenario_map: ip.Map,
                 goal: ip.Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 cost_factors: Dict[str, float] = None,
                 n_simulations: int = 5,
                 max_depth: int = 3,
                 store_results: str = 'final'):
        super(XAVIAgent, self).__init__(agent_id, initial_state, t_update, scenario_map, goal, view_radius, fps,
                                        cost_factors, n_simulations, max_depth, store_results)
        self._mcts = ip.MCTS(scenario_map,
                             n_simulations=n_simulations,
                             max_depth=max_depth,
                             store_results=store_results,
                             tree_type=XAVITree,
                             node_type=XAVINode)
        self._bn = XAVIBayesNetwork()

    def update_plan(self, observation: ip.Observation):
        """ Calls goal recognition and MCTS then updates the BN probabilities. """
        super(XAVIAgent, self).update_plan(observation)

        mcts_results = self._mcts.results
        self._bn.update(mcts_results)

    @property
    def bn(self) -> XAVIBayesNetwork:
        """ Return the Bayes network explainer of this agent. """
        return self._bn

    @property
    def mcts(self) -> ip.MCTS:
        """ Returns the MCTS search class of this agent. """
        return self._mcts
