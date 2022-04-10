import copy
from numbers import Number
from typing import Dict, Tuple, List, Optional, Any
import itertools
import numpy as np
import igp2 as ip
import networkx as nx
import random

import scipy.stats
from scipy.stats._distn_infrastructure import rv_frozen as rv_frozen


def softmax(x, axis: int = 0):
    """ Calculate a numerically stable version of softmax. """
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=0.5):
    """
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Copied from: https://epidemicsonnetworks.readthedocs.io/en/latest/_modules/EoN/auxiliary.html#hierarchy_pos

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx=0.2, vert_gap=0.2, vert_loc=0,
                       xcenter=0.5, rootpos=None,
                       leafpos=None, parent=None):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        """

        if rootpos is None:
            rootpos = {root: (xcenter, vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            rootdx = width / len(children)
            nextx = xcenter - width / 2 - rootdx / 2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G, child, leftmost + leaf_count * leafdx,
                                                             width=rootdx, leafdx=leafdx,
                                                             vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                                             xcenter=nextx, rootpos=rootpos, leafpos=leafpos,
                                                             parent=root)
                leaf_count += newleaves

            leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        #        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width / 2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node) == 0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node) == 1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width,
                                                  leafdx=width * 1. / leafcount,
                                                  vert_gap=vert_gap,
                                                  vert_loc=vert_loc,
                                                  xcenter=xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = leaf_vs_root_factor * leafpos[node][0] + \
                    (1 - leaf_vs_root_factor) * rootpos[node][0], leafpos[node][1]
        # pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1)
        #        for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos


def getindex(collection, index: str):
    """ Get a value from the collection with the given index represented as a string. """
    if ":" in index:
        parts = index.split(':')
        assert len(parts) == 2, f"Invalid slicing operation {index}"
        start, end = [None if x == "" else int(x) for x in parts]
        c = collection[start:end]
        if len(c) == 1:
            return c[0]
        return c

    if isinstance(collection, dict):
        return collection[index]
    else:
        return collection[int(index)]


class Normal:
    """ Univariate normal distributions extended with support for NoneType.
        Supports adding and multiplying normal PDFs."""

    def __init__(self, loc: Optional[float] = 0.0, scale: Optional[float] = 1.0):
        self._loc = loc
        self._scale = scale
        if loc is not None:
            self._norm = scipy.stats.norm(loc, scale)
        else:
            self._norm = None

    def __repr__(self):
        return str(self._norm)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot add Normal and {type(other)}")
        new_dist = Normal()
        new_dist._norm = copy.deepcopy(self._norm)
        if not isinstance(self._norm, list) or self._norm[0] != "+":
            new_dist._norm = ["+", new_dist._norm, other._norm]
        else:
            new_dist._norm.append(other._norm)
        return new_dist

    def __mul__(self, other):
        # If other is a Normal distribution than this will assume that the two distributions are independent
        new_dist = Normal()
        new_dist._norm = copy.deepcopy(self._norm)

        if isinstance(other, self.__class__):
            other_val = other._norm
        elif isinstance(other, Number):
            other_val = other

        if not isinstance(self._norm, list) or self._norm[0] != "*":
            new_dist._norm = ["*", new_dist._norm, other_val]
        else:
            new_dist._norm.append(other_val)

        return new_dist

    def __rmul__(self, other):
        return self.__mul__(other)

    def _eval(self, tree, x, cond):
        val = cond(tree, x)
        if val is not None:
            return val

        op = tree[0]
        if op == "+":
            val = 0.0
            for d in tree[1:]:
                val += self._eval(d, x, cond)
        elif op == "*":
            val = 1.0
            for d in tree[1:]:
                val *= self._eval(d, x, cond)
        return val

    def pdf(self, x: float) -> float:
        """ Evaluate the normal PDF at x. If the mean is None, then return 1.0 if x is None. """
        def cond(d_, x_):
            if d_ is None:
                return 1.0 if x_ is None else 0.0
            elif isinstance(d_, Number):
                return d_
            elif isinstance(d_, rv_frozen):
                return d_.pdf(x_) if x_ is not None else 0.0
        return self._eval(self._norm, x, cond)

    def cdf(self, x: float) -> float:
        def cond(d_, x_):
            if d_ is None:
                return 1.0 if x_ is None else 0.0
            elif isinstance(d_, Number):
                return d_
            elif isinstance(d_, rv_frozen):
                return d_.cdf(x_) if x_ is not None else 0.0
        return self._eval(self._norm, x, cond)

    def mean(self):
        """ Return the mean of the distribution. """
        def cond(d_, x_):
            if d_ is None:
                return 0.0
            elif isinstance(d_, Number):
                return d_
            elif isinstance(d_, rv_frozen):
                return d_.mean()
        return self._eval(self._norm, None, cond)

    def integrate(self) -> float:
        """ Return the definite integral of this distribution over the range (-inf, inf). """
        def cond(d_, x_):
            if d_ is None or isinstance(d_, rv_frozen):
                return 1.0
            elif isinstance(d_, Number):
                return d_
        return self._eval(self._norm, None, cond)

    def discretize(self,
                   low: float,
                   high: float,
                   bins: int,
                   norm: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Discretize the normal distribution into a probability mass function.
        Code copied from: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/continuous/discretize.py

        Args:
            low: Lower limit of the discretized distribution
            high: Higher limit of the discretized distribution
            bins: Number of elements in the discretized distribution
            norm: If true, then normalise the returning distribution

        Returns:
            A pair containing the discretized distribution and the bins of the distribution
        """
        step = (high - low) / bins
        discrete_values = [
            self.cdf(low + step / 2) - self.cdf(low)
        ]

        points = np.linspace(low + step, high - step, bins - 1)
        discrete_values.extend(
            [self.cdf(i + step / 2) - self.cdf(i - step / 2) for i in points]
        )

        dist = np.array(discrete_values)
        if norm:
            dist /= dist.sum()
        return dist, np.arange(low, high, step)

    @property
    def loc(self) -> Optional[float]:
        """ Return the mean of the distribution if it is not a composite distribution"""
        if isinstance(self._norm, rv_frozen) or self._norm is None:
            return self._loc

    @property
    def scale(self) -> Optional[float]:
        if isinstance(self._norm, rv_frozen) or self._scale is None:
            return self._scale


class Sample:
    """ Utility class to group samples of a sampling step in MCTS. """

    def __init__(self, samples: Dict[int, Tuple[ip.GoalWithType, ip.VelocityTrajectory]]):
        self.samples = samples

    def __eq__(self, other: "Sample"):
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        for aid, sample in other.samples.items():
            if aid not in self.samples:
                return False
            return self.samples[aid] == sample

    def __hash__(self):
        return hash(tuple(self.samples.items()))

    def __getitem__(self, item):
        return self.samples[item]

    def __repr__(self):
        return str(self.samples)

    @staticmethod
    def all_combinations(predictions: Dict[int, ip.GoalsProbabilities]) -> List["Sample"]:
        """ Get all possible sampling combinations from the goal predictions.

        Args:
            predictions: Goal predictions for agents.
        """
        def all_unqiue(tuples):
            aids = [a for a, _ in tuples]
            return len(aids) == len(set(aids))

        ret = []
        for aid, pred in predictions.items():
            for goal, trajectories in pred.all_trajectories.items():
                for trajectory in trajectories:
                    ret.append((aid, (goal, trajectory)))
        return [Sample(dict(x)) for x in itertools.combinations(ret, len(predictions)) if all_unqiue(x)]

    @property
    def agents(self) -> List[int]:
        """ The IDs of agents in the sample. """
        return list(self.samples)


if __name__ == '__main__':
    a = Normal(5, 1)
    b = Normal()
    c = 2 * a
    print(c)
    print((c * (a + c * c + a)).mean())
    print((a * 2 + b).mean())
    print((2 * a + c + 3 * b).integral())
