from typing import Dict, List, Any

import numpy as np

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
             show_progress: bool = False) -> float:
        """ Calculate the mean across the variables given the evidence, ignoring non-numeric values.

        Args:
            variables: List of variables to sum over for the mean
            evidence: Evidence to condition on
            virtual_evidence: Virtual evidences
            elimination_order: Order of variable elimination. If None, then computed automatically.
            joint: If True, returns a Joint Distribution over variables.
                Otherwise, return a dict of distributions over each of the variables.
            show_progress: Whether to show progress using tqdm
        """
        if joint:
            phi = self.query(variables, evidence, virtual_evidence, elimination_order, joint, show_progress)
            joint = phi.values
            values = np.prod(np.meshgrid(*[np.array(s, dtype=np.float64) for s in phi.state_names.values()]), axis=0)
            return np.nansum(values * joint)
        else:
            phi = self.query(variables, evidence, virtual_evidence, elimination_order, joint, show_progress)
            total = 0.0
            for var, factor in phi.items():
                total += np.nansum(factor.values * np.array(factor.state_names[var], dtype=np.float64))
            return total

    def entropy(self) -> float:
        # TODO: Replace with actual entropy calculation
        phi = self.query(["trajectory_1", "omega_2", "omega_3"], {"omega_1": "ChangeLaneLeft()"})
        joint = phi.values
        h = [-np.nansum(t * np.log2(t)) for t in joint.transpose(phi.variables.index("trajectory_1"))]