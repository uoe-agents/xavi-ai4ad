import copy
from functools import partial
from typing import Any, List, Callable, Dict

from xavi.cfg_util import *
from xavi.util import getindex

import igp2 as ip


class Token:
    """ Base-class for all tokens. Provides facilities to manage the passing of data when part of a
    production rule.

    Calling an instance object will set the data stored in the token. During the init of production rules,
    the data stored in this token will be used to set up a mapping from the arguments of hte left-hand side
    of the production to arguments of this token. You can also specify an index by having 'arg_name!index' passed
    to the call. This index can be any valid Python indexing or slicing.
    """

    def __init__(self, name: str, args: List[str]):
        """ Initialise a new token.

        Args:
            name: A descriptive name for the token
            args: The list of argument names this token can accept
        """
        self._name = name
        self._kwargs = {k: None for k in args}

    def __call__(self, *args, **kwargs) -> "Token":
        """ Sets the values of the arguments of the token. Returns a new token. """
        new_token = Token(self._name, list(self._kwargs))
        if args:
            new_token._kwargs = {k: v for k, v in zip(new_token._kwargs, args)}
        elif kwargs != {}:
            new_token._kwargs = {k: kwargs.get(k, None) for k in new_token._kwargs}
        return new_token

    def __repr__(self):
        return f"{self._name}" + f"[{','.join(self._kwargs)}]" if self.kwargs != {} else ""

    def __getitem__(self, item):
        return self._kwargs[item]

    def __setitem__(self, key, value):
        self._kwargs[key] = value

    def load_data(self, **data):
        self._kwargs = {k: data.get(k, None) for k in self._kwargs}

    @property
    def name(self) -> str:
        """ Name of the non-terminal. """
        return self._name

    @property
    def arguments(self) -> List[str]:
        """ A list of arguments this non-terminal can accept. """
        return list(self._kwargs)

    @property
    def kwargs(self) -> dict:
        """ Returns a mapping for the arguments of the non-terminal to placeholder values that will be replaced
         when expanded as part of a production. If the placeholder value is not a key in the data contained in the
         left-hand side non-terminal of a production rule, then the  the value will be used directly.
         If no such key exists then the actual value will be used directly. """
        return self._kwargs


class Terminal(Token):
    """ Represents a terminal token in a CFG. The terminal can take any number of arguments which can then be
    evaluated on the specified function to produce a final representation."""

    def __init__(self, name: Any, args: List[str], func: Callable[[Any], str] = to_str):
        """ Initialise a new terminal token with the given name, arguments and evaluation function.

        Args:
            name: A descriptive name for the token
            args: The list of argument names this token can accept
            func: Evaluation function to use
        """
        super(Terminal, self).__init__(name, args)
        self._f = func

    def __call__(self, *args, **kwargs) -> "Terminal":
        token = super(Terminal, self).__call__(*args, **kwargs)
        t = Terminal(token.name, token.arguments, self._f)
        t._kwargs = token._kwargs
        return t

    def eval(self) -> str:
        """ Evaluate the terminal's function on the keyword arguments stored in the token
        and return the final string representation. """
        return self._f(**self._kwargs)


class Nonterminal(Token):
    """ Represents a non-terminal token in the CFG that may take any number of arguments. Alias for a Token. """

    def __call__(self, *args, **kwargs) -> "Nonterminal":
        token = super(Nonterminal, self).__call__(*args, **kwargs)
        t = Nonterminal(token.name, token.arguments)
        t._kwargs = token._kwargs
        return t


class Production:
    """ A non-terminal token representing a production rule in the CFG.
     The production rule is an ordered list of productions (i.e. non-terminals) and terminals
     """

    def __init__(self,
                 name: Nonterminal,
                 production: List[Token],
                 applicability: Callable[[Any], bool] = None):
        """ Initialise a new non-terminal with the given name and production rules.

        Args:
            name: Left-hand side of the production rule.
            production: Production rule associated with the non-terminal.
            applicability: Optional function to evaluate the applicability of the rule given
                keyword arguments.
        """
        self._name = name
        self._production = []
        for p in production:
            if isinstance(p, str):
                self._production.append(p)
            else:
                duplicates = sum([p.name == r.name for r in self._production if isinstance(r, Token)])
                if duplicates > 0:
                    p._name = p._name + f"_{duplicates}"
                self._production.append(p)
        self._f = applicability

        self._args_map = {}
        for p in filter(lambda x: isinstance(x, Token), self._production):
            self._args_map[p.name] = copy.deepcopy(p.kwargs)

    def __repr__(self):
        return f"{repr(self._name)} := {' '.join(map(repr, self._production))}"

    def expand(self) -> List[Token]:
        """ Expand the production and set the relevant data fields to the specified values. """
        ret = []
        for t in self._production:
            if isinstance(t, str):
                ret.append(t)
                continue
            for arg, val in self._args_map[t.name].items():
                v = val
                index = None
                attr = None

                if isinstance(val, str):
                    if "!" in val:
                        val, index = val.split("!")
                    if "." in val:
                        val, attr = val.split(".")

                if val in self.lhs.kwargs:
                    v = self.lhs[val]
                    if index is not None:
                        if not isinstance(v, (list, dict, tuple)):
                            raise ValueError(f"Index operation given but "
                                             f"variable {val} in {self.lhs} is not a Collection!")
                        v = getindex(v, index)
                    elif attr is not None:
                        v = getattr(v, attr)
                t[arg] = v
            ret.append(t)
        return ret

    def applicable(self, **data) -> bool:
        """ Return true if the production rule is applicable given the keyword arguments. """
        if self._f is None:
            return True
        return self._f(**data)

    @property
    def lhs(self) -> Nonterminal:
        """ The left-hand side of the production rule. Same as the name of the production. """
        return self._name

    @property
    def rhs(self) -> List[Token]:
        """ The right-hand side of the production rule. """
        return self._production


class ContextFreeGrammar:
    """ A context-free grammar consisting of a set of productions and terminals. """

    def __init__(self, start: Nonterminal, rules: List[Production]):
        """ Initialise a new context-free grammar. The terminal and non-terminal vocabularies of the CFG are inferred
        automatically from the rules.

        Args:
            start: The starting non-terminal
            rules: The production rules.
        """
        self._s = start
        self._rules = rules
        self._check_rules()

    def get_productions(self, nt: Nonterminal, **data) -> List[Production]:
        """ Get all productions with the given non-terminal. The given data will be looaded into the returned
        productions.

        Args:
            nt: The non-terminal to check
        """
        productions = [r for r in self._rules
                       if r.lhs.name.split("_")[0] == nt.name.split("_")[0] and
                       list(data) == r.lhs.arguments and
                       r.applicable(**data)]
        return productions

    def expand(self, **data) -> str:
        """ Generate a sentential form recursively with terminals only from the CFG productions and the given data.

        Keyword Args:
            Arguments with names and values to be passed to the production rules.
        """
        self._s.load_data(**data)  # Load data into starting non-terminal
        productions = self.get_productions(self._s, **data)
        assert len(productions) == 1, f"Starting production {self._s} is not unique!"
        return self._expand(productions[0])

    def _expand(self, production: Production) -> str:
        ret = []
        for t in production.expand():
            if isinstance(t, str):
                ret.append(t)
            elif isinstance(t, Terminal):
                ret.append(t.eval())
            else:
                pr = self.get_productions(t, **t.kwargs)[0]  # For now select the first NT
                pr.lhs.load_data(**t.kwargs)
                ret.append(self._expand(pr))
        return " ".join(map(str, filter(lambda x: x != "", ret)))

    def _check_rules(self):
        """ Validate all production rules and their arguments. """
        for rule in self._rules:
            for t in rule.rhs:
                if isinstance(t, str):
                    continue
                for k, v in t.kwargs.items():
                    if not isinstance(v, str):
                        continue
                    arg = v
                    if "!" in v:
                        arg = v.split("!")[0]
                    elif "." in v:
                        arg = v.split(".")[0]
                    assert arg in rule.lhs.arguments, f"Argument {arg} for {t} not found " \
                                                      f"among left-hand side of {rule.lhs}"


class XAVIGrammar(ContextFreeGrammar):
    """ Define the grammar used to generate explanations for an XAVI agent. """

    def __init__(self, ego: ip.MCTSAgent,
                 frame: Dict[int, ip.AgentState],
                 scenario_map: ip.Map):
        """ Initialise a new grammar.

        Args:
            ego: The ego agent.
        """
        self.ego = ego
        self.frame = frame
        self.scenario_map = scenario_map

        prods = []

        # Define all non-terminals
        s = Nonterminal("S", ["cf", "effects", "causes"])
        action = Nonterminal("ACTION", ["agent", "omegas", "probability"])
        effects = Nonterminal("EFFECTS", ["outcome", "p_outcome", "effects"])
        causes = Nonterminal("CAUSES", ["causes"])
        cause = Nonterminal("CAUSE", ["cause"])
        macros = Nonterminal("MACROS", ["omegas"])
        comparisons = Nonterminal("COMPS", ["effects"])
        comparison = Nonterminal("COMP", ["effect"])
        outcome = Nonterminal("OUT", ["outcome", "p"])

        # Define all terminals
        agent = Terminal("Agent", ["agent"], agent_to_name)
        adverb = Terminal("Adverb", ["p"], p_to_adverb)
        macro = Terminal("Macro", ["macro"], self._ma2str)
        relation = Terminal("Rel", ["rew_diff"], diff_to_comp)
        reward = Terminal("Reward", ["r"], reward_to_str)
        out = Terminal("Outcome", ["o"], outcome_to_str)

        # Define production rules
        prods.append(Production(s, ["if", action(ego, "cf.omegas", None),
                                    "then it would", effects("cf.outcome", "cf.p_outcome", "effects"),
                                    "because", causes("causes"), "."
                                    ]))
        prods.append(Production(action,
                                [agent("agent"), adverb("probability"), macros("omegas")]))
        prods.append(Production(macros,
                                [macro("omegas")],
                                partial(is_type, "omegas", (ip.MCTSAction, ip.MacroAction))))
        prods.append(Production(macros,
                                [macros("omegas!0"), "then", macros("omegas!1:")],
                                partial(len_gt1, "omegas")))
        prods.append(Production(effects,
                                [""],
                                partial(none, "effects")))
        prods.append(Production(effects,
                                [outcome("outcome", "p_outcome"), comparisons("effects")]))
        prods.append(Production(comparisons,
                                [""],
                                partial(none, "effects")))
        prods.append(Production(comparisons,
                                [comparison("effects")],
                                partial(is_type, "effects", Effect)))
        prods.append(Production(comparisons,
                                [comparisons("effects!0"), "and", comparisons("effects!1:")],
                                partial(len_gt1, "effects")))
        # prods.append(Production(comparison,
        #                         [""],
        #                         partial(none, "effect.relation")))
        prods.append(Production(comparison,
                                ["with", relation("effect.relation"), reward("effect.reward")]))
        prods.append(Production(causes,
                                [""],
                                partial(none, "causes")))
        prods.append(Production(causes,
                                [cause("causes")],
                                partial(is_type, "causes", Cause)))
        prods.append(Production(causes,
                                [causes("causes!0"), "and", causes("causes!1:")],
                                partial(len_gt1, "causes")))
        prods.append(Production(cause,
                                [action("cause.agent", "cause.omegas", "cause.p_omegas")]))
        prods.append(Production(outcome,
                                [adverb("p"), out("outcome")]))

        super(XAVIGrammar, self).__init__(s, prods)

    def _ma2str(self, macro) -> str:
        return macro_to_str(self.ego.agent_id, self.frame, self.scenario_map, macro)
