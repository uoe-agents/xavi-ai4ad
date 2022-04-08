from typing import Dict, List, Any, Union, Tuple

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
             show_progress: bool = False) \
            -> Union[float, Dict[str, float]]:
        """ Calculate the mean across the variables given the evidence, ignoring non-numeric values.

        Args:
            variables: List of variables to sum over for the mean
            evidence: Evidence to condition on
            virtual_evidence: Virtual evidences
            elimination_order: Order of variable elimination. If None, then computed automatically.
            joint: If True, returns a Joint Distribution over variables.
                Otherwise, return a dict of distributions over each of the variables.
            show_progress: Whether to show progress using tqdm.

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

    def mean_differences(self,
                         variables: List[str],
                         factual: Dict[str, Any],
                         counterfactual: Dict[str, Any],
                         virtual_evidence: List[TabularCPD] = None,
                         elimination_order: str = 'MinFill',
                         joint: bool = False,
                         show_progress: bool = False
                         ) \
            -> Union[float, Dict[str, float]]:
        """ Calculate the expected differences in variable means resulting from switching to counterfactual conditions
        from the given factual conditions. If a variable did not change under the counterfactual, then it will have a
        difference of np.nan.

        Args:
            variables: Variables to compute differences for.
            factual: A dictionary of factual observations (evidence).
            counterfactual: A dictionary of counterfactual evidence.
            virtual_evidence: Virtual evidences
            elimination_order: Order of variable elimination. If None, then computed automatically.
            joint: If True, returns a Joint Distribution over variables.
                Otherwise, return a dict of distributions over each of the variables.
            show_progress: Whether to show progress using tqdm.

        Returns:
            If joint is true a single floating point number otherwise a dictionary of random variable means for each
            random variable.
        """
        if joint:
            f = self.mean(variables, factual, virtual_evidence, elimination_order, joint, show_progress)
            cf = self.mean(variables, counterfactual, virtual_evidence, elimination_order, joint, show_progress)
            diff = cf - f
            return np.nan if np.isclose(diff, 0.0) else diff
        else:
            ret = {}
            for v in variables:
                f = self.mean([v], factual, virtual_evidence, elimination_order, False, show_progress)
                cf = self.mean([v], counterfactual, virtual_evidence, elimination_order, False, show_progress)
                diff = cf[v] - f[v]
                ret[v] = np.nan if np.isclose(diff, 0.0) else diff
            return ret

    def rank_agent_influence(self,
                             evidence: Dict[str, Any] = None) \
            -> Dict[str, Dict[ip.VelocityTrajectory, float]]:
        """ Rank each non-ego agent by the extent of the effect their sampled trajectories have on the action choices
        of the ego vehicle. The score for each t in T is calculate as D_KL[P(Omega|T=t) || P(Omega)].

        Args:
            evidence: Optional evidence to condition on.

        Notes:
            An agent has larger influence on the ego if the total KL summed across the agent's trajectories is larger.
            A given trajectory for an agent is more likely to determine the actions of the ego if the KL of the
            trajectory is smaller, meaning the ego is more likely to choose the same actions more often.

        Returns:
            A dictionary for each agent (given as a trajectory r.v.) with the KL-divergence scores
            for each possible trajectory of that agent.
        """
        if evidence is None:
            evidence = {}

        trajectories = []
        omegas = []
        for node in self.model.nodes:
            if node.startswith("trajectory") and node not in evidence:
                trajectories.append(node)
            elif node.startswith("omega") and node not in evidence:
                omegas.append(node)

        diffs = {}
        for trajectory in trajectories:
            variables = omegas + [trajectory]
            phi = self.query(variables, evidence)
            var_order = [phi.variables.index(v) for v in variables]
            sum_axes = tuple(range(len(omegas)))

            p_tomega = phi.values.transpose(var_order)
            p_tomega = p_tomega[..., ~np.all(np.isclose(p_tomega, 0.0), axis=sum_axes)]
            p_t = p_tomega.sum(axis=tuple(range(len(omegas))), keepdims=True)
            p_omega = p_tomega.sum(axis=-1, keepdims=True)
            p_omega_t = p_tomega / p_t

            kl = np.nansum(p_omega_t * (np.log2(p_omega_t) - np.log2(p_omega)), axis=sum_axes)
            diff = np.clip(kl, a_min=0.0, a_max=None)
            diffs[trajectory] = {phi.no_to_name[trajectory][i]: diff[i] for i in np.argsort(diff)}
        return dict(sorted(diffs.items(), key=lambda x: -sum(x[1].values())))

    def most_likely_outcome(self,
                            evidence: Dict[str, Any],
                            virtual_evidence: List[TabularCPD] = None,
                            elimination_order: str = 'MinFill',
                            joint: bool = False,
                            show_progress: bool = False
                            ) \
            -> Tuple[Union[str, List[str]], float]:
        """ Calculate the most likely outcome given the observed evidence.

        Args:
            evidence: A dictionary of factual observations (evidence).
            virtual_evidence: Virtual evidences
            elimination_order: Order of variable elimination. If None, then computed automatically.
            joint: If True, returns a Joint Distribution over variables.
                Otherwise, return a dict of distributions over each of the variables.
            show_progress: Whether to show progress using tqdm.
        """
        variables = [cpd.variable for cpd in self.model.get_cpds() if cpd.variable.startswith("outcome")]
        phi = self.query(variables, evidence, virtual_evidence, elimination_order, joint, show_progress)

        if joint:
            amax = np.unravel_index(np.argmax(phi.values), phi.values.shape)
            max_prob = phi.values[amax]
            max_outcome = [comp for i, comp in zip(amax, phi.variables) if phi.state_names[comp][i]]
            return max_outcome, max_prob
        else:
            max_prob = -1
            max_outcome = None
            for outcome, factor in phi.items():
                p = factor.values[1]
                if p > max_prob:
                    max_prob, max_outcome = p, outcome
            return max_outcome, max_prob
