from typing import Any, List, Union

import re


class Terminal:
    """ Represents a terminal token in a CFG. """

    def __init__(self, token: Any):
        """ Initialise a new terminal with the given token. """
        self._token = token

    def __str__(self) -> str:
        return str(self._token)

    @property
    def lhs(self):
        return self._token


class Production:
    """ A non-terminal token representing a production rule in the CFG.
     The production rule is an ordered list of productions (i.e. non-terminals) and terminals
     """

    def __init__(self,
                 name: str,
                 production: List[Union[Terminal, "Production"]],
                 args: List[str]):
        """ Initialise a new non-terminal with the given name and production rules.

        Args:
            name: Name of the non-terminal. This need not be unique.
            production: Production rule associated with the non-terminal.
            args: A list of arguments this non-terminal can accept.
        """
        self._name = name
        self._production = production
        self._args = args

    @property
    def arguments(self) -> List[str]:
        """ A list of arguments this non-terminal can accept. """
        return self._args

    @property
    def lhs(self) -> str:
        """ The left-hand side of the production rule. Same as the name of the production. """
        return self._name

    @property
    def rhs(self) -> List[Union[Terminal, "Production"]]:
        """ The right-hand side of the production rule. """
        return self._production


class ContextFreeGrammar:
    """ A context-free grammar consisting of a set of productions and terminals. """

    def __init__(self, start: Production, rules: List[Production]):
        """ Initialise a new context-free grammar. The terminal and non-terminal vocabularies of the CFG are inferred
        automatically from the rules.

        Args:
            start: The starting non-terminal
            rules: The production rules.
        """
        self._s = start
        self._rules = rules

    def expand(self, **kwargs) -> str:
        """ Generate a sentential form recursively with terminals only from the CFG productions and the given data.

        Keyword Args:
            Arguments with names and values to be passed to the production rules.
        """
        return self._expand(self._start, **kwargs)

    def _expand(self, token: Production, **kwargs) -> Production:
        for t in token.rhs:
            productions = [r for r in self._rules if r.lhs == t.lhs]
            assert len(productions) > 0, f"No rule matched the token {t.lhs}!"

            # For now only select the first possible rule
            production = productions[0]
            pr_kwargs = {k: v for k, v in kwargs.items() if k in production}
            self._expand(productions[0], **pr_kwargs)

    @classmethod
    def from_string(cls, grammar: str) -> "ContextFreeGrammar":
        """ Parse a new grammar from a string of production rules. The starting rule must begin with the non-terminal
        'S' and be unique.
        """
        rules = []

        for i, line in enumerate(grammar.split("\n")):
            assert ":=" in line, f"No rule definition on line {i}!"
            lhs, rhs_str = line.split(":=")
            lhs, rhs_str = lhs.strip(), rhs_str.strip()
            rhs = []
            for elem in rhs_str.split(" "):
                elem = elem.strip()
                if elem[0] == elem[-1] == "'":
                    rhs.append(elem[1:-1])
                else:
                    match = re.match(r"^(\S+)\[(.*)\]$", elem)
                    if match is not None:
                        name = match.groups(0)
                        args = [a.strip() for a in match.groups(1)[1:-1].split(",")]

            new_rule = Production(lhs.strip(), rhs)
            rules.append(new_rule)

        start = [r for r in rules if r.lhs == "S"]
        assert len(start) == 1, f"Starting production is not unique!"
        start = start[0]

        return cls(start, rules)


if __name__ == '__main__':
    grammar_rules = """S[ego,omega_y,outcome_y,effects,causes] := 'if' ACTION[ego,omega_y,None] 'then we would' EFFECTS[outcome_y,effects] 'because' CAUSES[causes] '.' """
    cfg = ContextFreeGrammar.from_string(grammar_rules)