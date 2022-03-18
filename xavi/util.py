import copy
from typing import Dict, Tuple, List
import itertools
import numpy as np
import igp2 as ip
import networkx as nx
import random

import scipy.stats


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


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class Normal:
    """ Univariate normal distributions extended with support for NoneType. Supports adding and multiplying normal PDFs.
        Internally represented as a tree. """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self.loc = loc
        self.scale = scale
        if loc is not None:
            self._norm = scipy.stats.norm(loc, scale)
        else:
            self._norm = None

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
        new_dist = Normal()
        new_dist._norm = copy.deepcopy(self._norm)

        other_val = None
        if isinstance(other, self.__class__):
            other_val = other._norm
        elif isinstance(other, float):
            other_val = other
        else:
            raise ValueError(f"Cannot multiply Normal and {type(other)}")

        if not isinstance(self._norm, list) or self._norm[0] != "*":
            new_dist._norm = ["*", new_dist._norm, other_val]
        else:
            new_dist._norm.append(other_val)

        return new_dist

    def pdf(self, x: float):
        """ Evaluate the normal PDF at x. If the mean is None, then return 1.0 if x is None. """
        def _pdf(dists):
            if dists is None:
                return 1.0 if x is None else 0.0
            elif isinstance(dists, float):
                return dists
            elif not isinstance(dists, self.__class__) and hasattr(dists, "pdf"):
                return dists.pdf(x)

            op = dists[0]
            if op == "+":
                val = 0.0
                for d in dists[1:]:
                    val += _pdf(d)
            elif op == "*":
                val = 1.0
                for d in dists[1:]:
                    val *= _pdf(d)
            return val
        return _pdf(self._norm)


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
