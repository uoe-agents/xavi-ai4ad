import copy
from typing import Any, List, Union, Callable

from nl_helper import p_to_adverb, macro_to_str, agent_to_name
from util import identity


class Terminal:
    """ Represents a terminal token in a CFG. The right-hand side can take any number of arguments which are
    evaluated on the specified function."""

    def __init__(self, name: Any, func: Callable[[Any], str]):
        """ Initialise a new terminal with the given token and evaluation function. """
        self._name = name
        self._f = func

    def __repr__(self) -> str:
        return str(self._name)

    def __call__(self, *args, **kwargs):
        return self._f(**kwargs)

    @property
    def lhs(self):
        return self._name

    def rhs(self, **kwargs) -> str:
        """ Evaluate the terminal's function on the given keyword arguments and return a terminal token. """
        return self._f(**kwargs)


class Nonterminal:
    """ Represents a non-terminal token in the CFG that may take any number of arguments. """

    def __init__(self, name: str, args: List[str]):
        """ Initialise a new non-terminal. """
        self._name = name
        self._args_val = {k: None for k in args}

    def __repr__(self):
        return f"{self._name}[{','.join(self._args_val)}]"

    def __call__(self, *args, **kwargs) -> "Nonterminal":
        """ Sets the values of the arguments of the non-terminal. """
        if args:
            self._args_val = {k: v for k, v in zip(self._args_val, args)}
        elif kwargs != {}:
            self._args_val = {k: kwargs.get(k, None) for k in self._args_val}
        return self

    def __getitem__(self, item):
        return self._args_val[item]

    def __setitem__(self, key, value):
        self._args_val[key] = value

    @property
    def name(self) -> str:
        """ Name of the non-terminal. """
        return self._name

    @property
    def arguments(self) -> List[str]:
        """ A list of arguments this non-terminal can accept. """
        return list(self._args_val)

    @property
    def args_value(self) -> dict:
        """ Returns a mapping for the arguments of the non-terminal to placeholder values that will be replaced
         when expanded as part of a production. If the placeholder value is not a key in the data contained in the
         left-hand side non-terminal of a production rule, then the  the value will be used directly.
         If no such key exists then the actual value will be used directly. """
        return self._args_val


class Production:
    """ A non-terminal token representing a production rule in the CFG.
     The production rule is an ordered list of productions (i.e. non-terminals) and terminals
     """

    def __init__(self,
                 name: Nonterminal,
                 production: List[Union[Terminal, Nonterminal]],
                 applicability: Callable[[Any], bool] = lambda **kwargs: True):
        """ Initialise a new non-terminal with the given name and production rules.

        Args:
            name: Left-hand side of the production rule.
            production: Production rule associated with the non-terminal.
            applicability: Optional function to evaluate the applicability of the rule given
                keyword arguments.
        """
        self._name = name
        self._production = production
        self._args_map = {}
        for p in filter(lambda x: isinstance(x, Nonterminal), production):
            self._args_map[p.name] = copy.deepcopy(p.args_value)
        self._f = applicability

    def __repr__(self):
        return f"{repr(self._name)} := {' '.join(map(repr, self._production))}"

    def expand(self) -> List[Union[Terminal, Nonterminal]]:
        """ Expand the production and set the relevant data fields to the specified values. """
        ret = []
        for t in self._production:
            if isinstance(t, str):
                ret.append(t)
                continue
            for arg, val in self._args_map[t.name].items():
                t[arg] = self.lhs[val] if val in self.lhs else val
            ret.append(t)
        return ret

    @property
    def lhs(self) -> Nonterminal:
        """ The left-hand side of the production rule. Same as the name of the production. """
        return self._name

    @property
    def rhs(self) -> List[Union[Terminal, Nonterminal]]:
        """ The right-hand side of the production rule. """
        return self._production

    @property
    def applicable(self) -> bool:
        """ Return true if the production rule is applicable given the keyword arguments stored
        in the left-hand side non-terminal. """
        return self._f(**self.lhs.args_value)


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
        self._terminals = []
        self._nonterminals = []

    def get_productions(self, nt: Nonterminal, **data) -> List[Production]:
        """ Get all productions with the given non-terminal.

        Args:
            nt: The non-terminal to check
            **data: Optional dictionary of keyword arguments containing data that may be used to check for
                specific applicability conditions of a production rule.
        """
        productions = [r for r in self._rules if r.lhs == nt and r.applicable]
        return productions

    def expand(self, **data) -> str:
        """ Generate a sentential form recursively with terminals only from the CFG productions and the given data.

        Keyword Args:
            Arguments with names and values to be passed to the production rules.
        """
        self._s(**data)  # Load data into starting non-terminal
        productions = self.get_productions(self._s)
        assert len(productions) == 1, f"Starting production is not unique {self._s}!"
        return self._expand(productions[0])

    def _expand(self, production: Production) -> str:
        rhs = production.expand()

        ret = ""
        for t in rhs:
            if isinstance(t, str):
                ret += t
            elif isinstance(t, Terminal):
                ret += t()
            else:
                pr = self.get_productions(t)[0]  # For now select the first NT
                ret += self._expand(pr)
        return ret


if __name__ == '__main__':
    prods = []

    s = Nonterminal("S", ["ego", "cf", "effects", "causes"])
    action = Nonterminal("ACTION", ["agent", "omegas", "cf"])
    effects = Nonterminal("EFFECTS", ["outcome", "p_outcome", "effects"])
    causes = Nonterminal("CAUSES", ["causes"])
    macros = Nonterminal("MACROS", ["omegas"])

    agent = Terminal("Agent", agent_to_name)
    adverb = Terminal("Adverb", p_to_adverb)
    macro = Terminal("Macro", macro_to_str)

    prods.append(Production(s, ["if",
                                action("ego", "cf", None),
                                "then we would",
                                effects("effects", "cf"),
                                "because",
                                causes("causes")]))
    prods.append(Production(action, [agent("ego"), adverb("p"), "take", macros("omegas")]))
    prods.append(Production(macros, [macro], lambda omegas: len(omegas) == 1))
    prods.append(Production(macros, [macros, "then", macros], lambda omegas: len(omegas) > 1))

    cfg = ContextFreeGrammar(s, prods)
    text = cfg.expand(ego=1, cf=[2, 3, 4], effects=[5, 6, 7, 8], causes=[10, 11])
