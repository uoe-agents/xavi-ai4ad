import pickle
from collections import namedtuple
from typing import Dict, List
import logging

import igp2 as ip
import numpy as np

from xavi.bayes_network import XAVIBayesNetwork
from xavi.node import XAVINode
from xavi.tree import XAVITree
from xavi.inference import XAVIInference
from xavi.cfg import XAVIGrammar
from xavi.cfg_util import Counterfactual, Effect, Cause


logger = logging.getLogger(__name__)

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
                 reward_factors: Dict[str, float] = None,
                 n_simulations: int = 5,
                 max_depth: int = 3,
                 store_results: str = 'final'):
        super(XAVIAgent, self).__init__(agent_id, initial_state, t_update, scenario_map, goal, view_radius, fps,
                                        cost_factors, reward_factors, n_simulations, max_depth, store_results)
        self._mcts = ip.MCTS(scenario_map,
                             n_simulations=n_simulations,
                             max_depth=max_depth,
                             store_results=store_results,
                             tree_type=XAVITree,
                             node_type=XAVINode)
        self._bayesian_network = XAVIBayesNetwork()
        self._inference = None
        self._grammar = None

    def update_plan(self, observation: ip.Observation):
        """ Calls goal recognition and MCTS then updates the BN probabilities. """
        super(XAVIAgent, self).update_plan(observation)

        mcts_results = self._mcts.results
        if isinstance(mcts_results, ip.MCTSResult):
            mcts_results = ip.AllMCTSResult()
            mcts_results.add_data(self._mcts.results)
        self._bayesian_network.update(mcts_results)
        self._inference = XAVIInference(self._bayesian_network.bn)
        self._grammar = XAVIGrammar(self, observation.frame, observation.scenario_map)

        logger.info(f"Generating explanations for factual action: {self.current_macro}")
        self.explain_all_actions()

    def explain_all_actions(self):
        for action in self.bayesian_network.variables["omega_1"]
            if action == str(self.current_macro):
                continue
            logger.log(f"\t{action}")
            explanation = self.explain_action({"omega_1": action}, n_causes=1, n_effects=1)
            logger.log(f"\t{explanation}")

    def explain_action(self,
                       counterfactual: Dict[str, str],
                       n_effects: int = 1,
                       n_causes: int = 1):
        """ Generate a contrastive explanation for the given counterfactual question.

        Args:
            counterfactual: Dictionary mapping action steps (omegas) to counterfactual actions represented as strings.
            n_effects: Number of effects to include in the explanation.
            n_causes: Number of causes to include in the explanation.
        """
        factual = {f"omega_{d}": str(ma) for d, ma in enumerate(self._macro_actions, 1)}

        # Get counterfactual outcome
        cf_omegas = [self._bayesian_network.macro_actions[a] for k, a in sorted(counterfactual.items())]
        cf_outcome, cf_p_outcome = self._inference.most_likely_outcome(evidence=counterfactual)
        if len(cf_omegas) == 1:
            cf_omegas = cf_omegas[0]
        cf = Counterfactual(cf_omegas, cf_outcome, cf_p_outcome)

        # Get effects of choosing counterfactual
        effects = []
        if cf_outcome == "outcome_coll":
            node_key = ("Root", ) + tuple(counterfactual.values())
            colliding_agents = self.bayesian_network.tree.collision_from_node(node_key)
            for colliding_agent in colliding_agents:
                effects.append(Effect(colliding_agent, None))
        elif cf_outcome == "outcome_done":
            variables = [var for var in self._bayesian_network.variables if var.startswith("reward")]
            rew_diffs = self._inference.mean_differences(variables, factual, counterfactual)
            for r, r_diff in sorted(rew_diffs.items(), key=lambda item: -np.abs(item[1])):
                if len(effects) == n_effects:
                    break
                effects.append(Effect(r_diff, r))
        elif cf_outcome == "outcome_dead":
            effects.append(None)
        if len(effects) == 1:
            effects = effects[0]

        agent_influences = self._inference.rank_agent_influence(counterfactual)
        causes = []
        if cf_outcome != "outcome_dead" or n_causes > 1:
            for traj_aid, trajectories in agent_influences.items():
                if len(causes) == n_causes:
                    break
                if len(trajectories) > 1 and np.isclose(sum(trajectories.values()), 0.0):
                    # We ignore vehicles whose actions have no effect on the ego.
                    #  We check if there are more than 1 trajectory since having a single trajectory means that
                    #  the KL-divergence will be zero anyway
                    continue
                aid = int(traj_aid.split("_")[-1])
                predictions = self._bayesian_network.tree.predictions[aid]
                trajectory, kld = list(trajectories.items())[0]
                plan = None
                goal = None
                for g, t in predictions.all_trajectories.items():
                    if trajectory in t:
                        plan = predictions.all_plans[g][t.index(trajectory)]
                        goal = g
                        break
                agent = self._bayesian_network.tree.agents[aid]
                plan = [p for p in plan if not isinstance(p, ip.Continue)]
                if len(plan) == 1:
                    plan = plan[0]
                p_omegas = self._bayesian_network.p_t(aid, goal, trajectory)
                causes.append(Cause(agent, plan, p_omegas))
            if len(causes) == 1:
                causes = causes[0]
        else:
            causes = None

        data = {
            "cf": cf,
            "effects": effects,
            "causes": causes
        }
        return self._grammar.expand(**data)

    @property
    def bayesian_network(self) -> XAVIBayesNetwork:
        """ Return the Bayes network explainer of this agent. """
        return self._bayesian_network

    @property
    def mcts(self) -> ip.MCTS:
        """ Returns the MCTS search class of this agent. """
        return self._mcts
