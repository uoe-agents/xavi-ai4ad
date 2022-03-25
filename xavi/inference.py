from typing import Dict, List, Any, Union

import numpy as np
import igp2 as ip

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class XAVIInference(VariableElimination):
    """ Variable elimination-based inference for Bayesian Networks with extra convenience functionalities. """

    def mean(self,
             variables: List[str],
             evidence: Dict[str, Any] = None,
             virtual_evidence: List[TabularCPD] = None,
             elimination_order: str = 'MinFill',
             joint: bool = True,
             show_progress: bool = False) -> Union[float, Dict[str, float]]:
        """ Calculate the mean across the variables given the evidence, ignoring non-numeric values.

        Args:
            variables: List of variables to sum over for the mean
            evidence: Evidence to condition on
            virtual_evidence: Virtual evidences
            elimination_order: Order of variable elimination. If None, then computed automatically.
            joint: If True, returns a Joint Distribution over variables.
                Otherwise, return a dict of distributions over each of the variables.
            show_progress: Whether to show progress using tqdm

        Returns:
            A float if joint is True, otherwise a dictionary of means.
        """
        phi = self.query(variables, evidence, virtual_evidence, elimination_order, joint, show_progress)

        if joint:
            joint = phi.values
            values = np.prod(np.meshgrid(*[np.array(s, dtype=np.float64) for s in phi.state_names.values()]), axis=0)
            return np.nansum(values * joint)
        else:
            total = {}
            for var, factor in phi.items():
                total[var] = np.nansum(factor.values * np.array(factor.state_names[var], dtype=np.float64))
            return total

    def rank_agent_influence(self) -> Dict[str, Dict[ip.VelocityTrajectory, float]]:
        """ Rank each non-ego agent by the extent of the effect their sampled trajectories have on the action choices
        of the ego vehicle. The score for each t in T is calculated as E_Omega [(P(Omega|T=t) - P(Omega)) ** 2].

        Notes:
            An agent has larger influence on the ego if the total variance across the agent's trajectories is larger.
            A given trajectory for an agent is more likely to determine the actions of the ego if the variance of the
            trajectory is smaller, meaning the ego is more likely to choose the same actions more often.

        Returns:
            A dictionary for each agent (given as a trajectory r.v.) with variance scores for each possible trajectory
            of that agent.
        """
        trajectories = []
        omegas = []
        for node in self.model.nodes:
            if node.startswith("trajectory"):
                trajectories.append(node)
            elif node.startswith("omega"):
                omegas.append(node)

        variances = {}
        for trajectory in trajectories:
            variables = omegas + [trajectory]
            phi = self.query(variables)
            var_order = [phi.variables.index(v) for v in variables]
            sum_axes = tuple(range(len(omegas)))

            p_tomega = phi.values.transpose(var_order)
            # Drop trajectory for which all actions have zero probability
            p_tomega = p_tomega[..., ~np.all(np.isclose(p_tomega, 0.0), axis=sum_axes)]
            p_t = p_tomega.sum(axis=tuple(range(len(omegas))), keepdims=True)
            p_omega = p_tomega.sum(axis=-1, keepdims=True)
            p_omega_t = p_tomega / p_t
            var = np.sum(p_omega * (p_omega_t - p_omega) ** 2, axis=sum_axes)
            variances[trajectory] = {phi.no_to_name[trajectory][i]: var[i] for i in np.argsort(var)}
        return variances
