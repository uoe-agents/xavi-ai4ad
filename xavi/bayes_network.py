import xavi
import numpy as np


class XAVIBayesNetwork:
    """ Create a Bayesian network model from a MCTS search tree. """

    def __init__(self, tree: xavi.XAVITree):
        self._tree = tree

        self.p_sampling = {}  # Bold, capital T in paper
        for aid, pred in self._tree.predictions.items():
            self.p_sampling[aid] = {}
            for goal, trajectories in pred.all_trajectories.items():
                p_goal = pred.goals_probabilities[goal]
                p_trajectories = np.array(pred.trajectories_probabilities[goal])
                self.p_sampling[aid][goal] = p_goal * p_trajectories



    @property
    def tree(self) -> xavi.XAVITree:
        """ The MCTS search tree associated with this network. """
        return self._tree
