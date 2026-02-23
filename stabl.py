#!/usr/bin/python3

import contextlib
import subprocess
import string
import sys


from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import (
    Optional,
    Generic,

    List,
    Set,
    Dict,
    Tuple,

    TypeVar,
    Callable,

    NoReturn,
)

import typing

T = TypeVar("T")


def nth_permutation(n: int, basket: str) -> str:
    b = len(basket)
    s = []

    while True:
        s.append(basket[n % b])
        n = n // b - 1

        if n < 0:
            break

    return "".join(reversed(s))


## Debug


Location = Tuple[str, int, int]


class Error(Exception):
    def __init__(self, location: Optional[Location], message: str):
        self.location = location
        self.message = message

        super().__init__(self.__str__())


## Lexer


@dataclass(frozen=True)
class Token(ABC):
    location: Location


@dataclass(frozen=True)
class Token_EOF(Token):
    pass


@dataclass(frozen=True)
class Token_Dot(Token):
    pass


@dataclass(frozen=True)
class Token_Plus(Token):
    pass


@dataclass(frozen=True)
class Token_Minus(Token):
    pass


@dataclass(frozen=True)
class Token_Minus_Minus(Token):
    pass


@dataclass(frozen=True)
class Token_Star(Token):
    pass


@dataclass(frozen=True)
class Token_Slash(Token):
    pass


@dataclass(frozen=True)
class Token_Bang(Token):
    pass


@dataclass(frozen=True)
class Token_At(Token):
    pass


@dataclass(frozen=True)
class Token_Colon(Token):
    pass


@dataclass(frozen=True)
class Token_If(Token):
    pass


@dataclass(frozen=True)
class Token_While(Token):
    pass


@dataclass(frozen=True)
class Token_LBracket(Token):
    pass


@dataclass(frozen=True)
class Token_RBracket(Token):
    pass


@dataclass(frozen=True)
class Token_LBrace(Token):
    pass


@dataclass(frozen=True)
class Token_RBrace(Token):
    pass


@dataclass(frozen=True)
class Token_Real(Token):
    value: float


@dataclass(frozen=True)
class Token_Symbol(Token):
    value: str


@dataclass(frozen=True)
class Token_String(Token):
    value: str


@dataclass(frozen=True)
class Token_Import(Token):
    pass


@dataclass(frozen=True)
class Token_Print(Token):
    pass


@dataclass(frozen=True)
class Token_Do(Token):
    pass


@dataclass(frozen=True)
class Token_Unit(Token):
    pass


@dataclass(frozen=True)
class Token_Type(Token):
    pass


@dataclass(frozen=True)
class Token_Type_Real(Token):
    pass


@dataclass(frozen=True)
class Token_Forall(Token):
    pass


class Lexer:
    def __init__(self, text: str, file: str) -> None:
        self.text = iter(text)
        self.current = None

        self.file = file
        self.line = 1
        self.column = 0

        self.advance()


    @property
    def location(self) -> Location:
        return self.file, self.line, self.column


    def advance(self):
        self.current = next(self.text, None)

        if self.current == "\n":
            self.line += 1
            self.column = 0
        else:
            self.column += 1


    def advance_with(self, token) -> Token:
        self.advance()
        return token


    def next(self) -> Token:
        while self.current and self.current in string.whitespace:
            self.advance()

        if self.current in ("#", "\\"):
            while self.current is not None and self.current != "\n":
                self.advance()
            return self.next()

        buffer = ""
        location: Location = self.location

        if self.current is None:
            return self.advance_with(Token_EOF(location))

        if self.current == ".":
            return self.advance_with(Token_Dot(location))

        if self.current == "+":
            return self.advance_with(Token_Plus(location))

        if self.current == "-":
            self.advance()

            if self.current == "-":
                return self.advance_with(Token_Minus_Minus(location))

            return Token_Minus(location)

        if self.current == "*":
            return self.advance_with(Token_Star(location))

        if self.current == "/":
            return self.advance_with(Token_Slash(location))

        if self.current == "!":
            return self.advance_with(Token_Bang(location))

        if self.current == "@":
            return self.advance_with(Token_At(location))

        if self.current == ":":
            return self.advance_with(Token_Colon(location))

        if self.current == "[": # ]
            return self.advance_with(Token_LBracket(location))

        if self.current == "]":
            return self.advance_with(Token_RBracket(location))

        if self.current == "{": # }
            return self.advance_with(Token_LBrace(location))

        if self.current == "}":
            return self.advance_with(Token_RBrace(location))

        if self.current == '"':
            self.advance()

            while self.current and self.current != '"':
                buffer += self.current
                self.advance()

            if not self.current:
                raise Error(location, "unterminated string-literal")

            self.advance()

            return Token_String(location, buffer)

        if self.current.isdigit():
            if self.current == '0':
                buffer += self.current
                self.advance()

                if self.current in 'xX':
                    buffer += self.current
                    self.advance()

                    while self.current and self.current in string.hexdigits:
                        buffer += self.current
                        self.advance()

                    try:
                        return Token_Real(location, float(int(buffer, 16)))
                    except ValueError:
                        raise Error(location, "malformed number")

                if self.current in 'oO':
                    buffer += self.current
                    self.advance()

                    while self.current and self.current in string.octdigits:
                        buffer += self.current
                        self.advance()

                    try:
                        return Token_Real(location, float(int(buffer, 8)))
                    except ValueError:
                        raise Error(location, "malformed number")

                if self.current in 'bB':
                    buffer += self.current
                    self.advance()

                    while self.current and self.current in "01":
                        buffer += self.current
                        self.advance()

                    try:
                        return Token_Real(location, float(int(buffer, 2)))
                    except ValueError:
                        raise Error(location, "malformed number")

            integer = True

            while self.current and self.current in string.digits:
                buffer += self.current
                self.advance()

            if self.current == ".":
                integer = False

                buffer += self.current
                self.advance()

            while self.current and self.current in string.digits:
                buffer += self.current
                self.advance()

            if self.current in "eE":
                integer = False

                buffer += self.current
                self.advance()

                if self.current in "+-":
                    buffer += self.current
                    self.advance()

                while self.current and self.current in string.digits:
                    buffer += self.current
                    self.advance()

            try:
                return Token_Real(location, float(buffer))
            except ValueError:
                raise Error(location, "malformed number")

        if self.current.isalpha() or self.current in "_":
            while self.current and (self.current.isalnum() or self.current in "!'-?_"):
                buffer += self.current
                self.advance()

            if buffer == "Import":
                return Token_Import(location)

            if buffer == "Print":
                return Token_Print(location)

            if buffer == "Do":
                return Token_Do(location)

            if buffer == "If":
                return Token_If(location)

            if buffer == "While":
                return Token_While(location)

            if buffer == "Unit":
                return Token_Unit(location)

            if buffer == "Type":
                return Token_Type(location)

            if buffer == "Real":
                return Token_Type_Real(location)

            if buffer == "forall":
                return Token_Forall(location)

            return Token_Symbol(location, buffer)

        raise Error(location, "unexpected character")


## Types


Substitution = Dict[int, 'Type']


def compose_substitution(s1: Substitution, s2: Substitution) -> Substitution:
    return {k: v.apply(s2) for k, v in s1.items()} | s2


@dataclass(frozen=True)
class Type(ABC):
    location: Location

    @abstractmethod
    def free_type_variables(self) -> Set[int]: ...

    @abstractmethod
    def free_type_variables_exclude_effect(self) -> Set[int]: ...

    @abstractmethod
    def apply(self, s: Substitution) -> 'Type': ...


@dataclass(frozen=True)
class Type_Var(Type):
    # 'id' is positive for bound type variables
    # 'id' is negative for free type variables
    id: int

    def free_type_variables(self) -> Set[int]:
        return {self.id} if self.free else set()

    def free_type_variables_exclude_effect(self) -> Set[int]:
        return self.free_type_variables()

    def apply(self, s: Substitution) -> Type:
        return s.get(self.id, self)

    @property
    def bound(self) -> bool:
        return self.id >= 0

    @property
    def free(self) -> bool:
        return self.id < 0

    def __str__(self) -> str:
        if self.free:
            return "ɑ{}".format(-1 - self.id)
        else:
            return "β{}".format(self.id)


_type_var_counter = -1


def fresh_free_type_variable(location: Location) -> Type_Var:
    global _type_var_counter

    t = Type_Var(location, _type_var_counter)

    _type_var_counter -= 1

    return t


@dataclass(frozen=True)
class Type_Real(Type):
    def free_type_variables(self) -> Set[int]:
        return set()

    def free_type_variables_exclude_effect(self) -> Set[int]:
        return self.free_type_variables()

    def apply(self, s: Substitution) -> 'Type_Real':
        return self

    def __str__(self) -> str:
        return "Real"


@dataclass(frozen=True)
class Type_Stack(Type):
    tail: Type
    head: Type

    def free_type_variables(self) -> Set[int]:
        tail = self.tail.free_type_variables()
        head = self.head.free_type_variables()
        return tail | head

    def free_type_variables_exclude_effect(self) -> Set[int]:
        tail = set()
        head = set()

        if not isinstance(self.tail, Type_Effect):
            tail = self.tail.free_type_variables_exclude_effect()

        if not isinstance(self.head, Type_Effect):
            head = self.head.free_type_variables_exclude_effect()

        return tail | head

    def apply(self, s: Substitution) -> 'Type_Stack':
        return Type_Stack(self.location, self.tail.apply(s), self.head.apply(s))

    def __str__(self) -> str:
        if isinstance(self.head, Type_Stack):
            return "{} ({})".format(self.tail, self.head)
        else:
            return "{} {}".format(self.tail, self.head)


def stack_from_list(types_0: Type, *types: Type) -> Type:
    result = types_0

    for t in types:
        result = Type_Stack(result.location, result, t)

    return result


@dataclass(frozen=True)
class Type_Effect(Type):
    lhs: Type
    rhs: Type

    def free_type_variables(self) -> Set[int]:
        lhs = self.lhs.free_type_variables()
        rhs = self.rhs.free_type_variables()
        return lhs | rhs

    def free_type_variables_exclude_effect(self) -> Set[int]:
        lhs = set()
        rhs = set()

        if not isinstance(self.lhs, Type_Effect):
            lhs = self.lhs.free_type_variables_exclude_effect()

        if not isinstance(self.rhs, Type_Effect):
            rhs = self.rhs.free_type_variables_exclude_effect()

        return lhs | rhs

    def apply(self, s: Substitution) -> 'Type_Effect':
        return Type_Effect(self.location, self.lhs.apply(s), self.rhs.apply(s))

    def __str__(self) -> str:
        return "[{} -- {}]".format(self.lhs, self.rhs)


def compose_effect(e1: Type_Effect, e2: Type_Effect) -> Tuple[Type_Effect, Substitution]:
  s = unify(e1.rhs, e2.lhs)

  lhs = e1.lhs.apply(s)
  rhs = e2.rhs.apply(s)

  return Type_Effect(e1.location, lhs, rhs), s


@dataclass(frozen=True)
class Scheme:
    vs: Set[int]
    t: Type

    def free_type_variables(self) -> Set[int]:
        return self.t.free_type_variables() - self.vs

    def instantiate(self) -> Type:
        s: Substitution = {}

        for v in self.vs:
            s[v] = fresh_free_type_variable(self.t.location)

        return self.t.apply(s)

    def apply(self, s: Substitution) -> 'Scheme':
        return Scheme(self.vs, self.t.apply({k: v for k, v in s.items() if k not in self.vs}))

    @property
    def is_polymorphic(self) -> bool:
        return bool(self.vs)

    def __str__(self) -> str:
        return "∀ {}. {}".format(" ".join(("β{}".format(v) for v in self.vs)), self.t)


def generalize(t: Type) -> Scheme:
    return Scheme(t.free_type_variables(), t)


COLORS = [
    "\033[35m",
    "\033[36m",
]

RESET = "\033[0m" #]]]


def color_wrap(text: str, depth: int) -> str:
    return f"{COLORS[depth % len(COLORS)]}{text}{RESET}"



def align_strings(s1: str, s2: str) -> Tuple[str, str]:
    max_len = max(len(s1), len(s2))

    return s1.ljust(max_len), s2.ljust(max_len)


def align_types_impl(t1: Type, t2: Type, depth: int = 0) -> Tuple[str, str]:
    if isinstance(t1, Type_Stack) and isinstance(t2, Type_Stack):
        tail1, tail2 = align_types_impl(t1.tail, t2.tail, depth + 1)
        head1, head2 = align_types_impl(t1.head, t2.head, depth)

        return align_strings(f"{tail1} {head1}", f"{tail2} {head2}")

    if isinstance(t1, Type_Effect) and isinstance(t2, Type_Effect):
        lhs1, lhs2 = align_types_impl(t1.lhs, t2.lhs, depth + 1)
        rhs1, rhs2 = align_types_impl(t1.rhs, t2.rhs, depth + 1)

        return align_strings(f"[{lhs1} -- {rhs1}]", f"[{lhs2} -- {rhs2}]")

    s1 = color_wrap(str(t1), depth)
    s2 = color_wrap(str(t2), depth)

    return align_strings(s1, s2)


def align_types(t1: Type, t2: Type) -> Tuple[str, str]:
    s1, s2 = align_types_impl(t1, t2)

    return s1.strip(), s2.strip()


Unification_Stack = List[Tuple[Type, Type, str]]


def unification_stack_trace(stack: Unification_Stack) -> str:
    trace = ""

    for i, (t1, t2, path) in enumerate(stack):
        if i > 0:
            trace += "\n\n"

        trace += "  while unifying {},\n".format(path)

        s1, s2 = align_types(t1, t2)

        trace += "    a: '{}'\n".format(s1)
        trace += "    b: '{}'".format(s2)

    # trace += "\n\n  the unification failed.".format()

    return trace


def unify(t1: Type, t2: Type, stack: Optional[Unification_Stack] = None, path = "two types") -> Substitution:
    if stack is None:
        stack = []

    stack = stack + [(t1, t2, path)]

    if isinstance(t1, Type_Var):
        return bind(t1, t2, stack)

    if isinstance(t2, Type_Var):
        return bind(t2, t1, stack)

    if isinstance(t1, Type_Stack) and isinstance(t2, Type_Stack):
        s1 = unify(t1.tail, t2.tail, stack, "their tail")
        s2 = unify(t1.head.apply(s1), t2.head.apply(s1), stack, "their head")
        return compose_substitution(s1, s2)

    if isinstance(t1, Type_Effect) and isinstance(t2, Type_Effect):
        s1 = unify(t1.lhs, t2.lhs, stack, "their left-hand side")
        s2 = unify(t1.rhs.apply(s1), t2.rhs.apply(s1), stack, "their right-hand side")
        return compose_substitution(s1, s2)

    if type(t1) == type(t2):
        return {}

    raise Error(t1.location, f"\n{unification_stack_trace(stack)}\n\n  an error occured:\n    cannot unify '{t1}' with '{t2}'.")


def bind(t1: Type_Var, t2: Type, stack: Unification_Stack) -> Substitution:
    if t1 == t2:
        return {}

    if t1.id in t2.free_type_variables():
        raise Error(t1.location, f"\n{unification_stack_trace(stack)}\n\n  an error occured:\n    cannot unify '{t1}' with '{t2}'.")

    return {t1.id: t2}


## Parser


@dataclass(frozen=True)
class Tree(ABC):
    location: Location


@dataclass(frozen=True)
class Tree_Add(Tree):
    pass


@dataclass(frozen=True)
class Tree_Sub(Tree):
    pass


@dataclass(frozen=True)
class Tree_Mul(Tree):
    pass


@dataclass(frozen=True)
class Tree_Div(Tree):
    pass


@dataclass(frozen=True)
class Tree_Let(Tree):
    name: str


@dataclass(frozen=True)
class Tree_Def(Tree):
    name: str
    t: Scheme


@dataclass(frozen=True)
class Tree_If(Tree):
    pass


@dataclass(frozen=True)
class Tree_While(Tree):
    pass


@dataclass(frozen=True)
class Tree_Block(Tree):
    id: int
    body: List[Tree]

    name: Optional[str] = None

    captured_lets: Set[str] = field(default_factory=set)
    captured_defs: Set[str] = field(default_factory=set)

    @property
    def is_closure(self) -> bool:
        return bool(self.captured_lets or self.captured_defs)


@dataclass(frozen=True)
class Tree_Real(Tree):
    value: float


@dataclass(frozen=True)
class Tree_Unit(Tree):
    pass


@dataclass(frozen=True)
class Tree_Symbol(Tree):
    value: str


@dataclass(frozen=True)
class Tree_Print(Tree):
    pass


@dataclass(frozen=True)
class Tree_Do(Tree):
    pass


@dataclass(frozen=True)
class Tree_Type(Tree):
    t: Scheme


def tree_print(s: Tree, depth=0):
    print("  " * depth, type(s).__name__, sep="", end="")

    match s:
        case Tree_Let(name=name):
            print(" {}".format(name))
        case Tree_Def(name=name, t=t):
            print(" {}: {}".format(name, t))
        case Tree_Block(body=body):
            # print(f" id={s.id}")
            print()
            for s in body:
                tree_print(s, depth=depth + 1)
        case Tree_Real(value=value):
            print(" {}".format(value))
        case Tree_Symbol(value=value):
            print(" {}".format(value))
        case Tree_Type(t=t):
            print(" {}".format(t))
        case _:
            print()


class Parser:
    def __init__(self, text: str, file: str, start_block_id: int = 0, importing: Optional[Set[str]] = None):
        self.lexer = Lexer(text, file)
        self.advance()

        self.next_location: List[Location] = []

        self.block_id = start_block_id

        self.importing = importing or set()
        self.importing.add(file)

    @property
    def location(self) -> Location:
        if self.next_location:
            return self.next_location.pop()

        return self.current.location

    def advance(self):
        self.current = self.lexer.next()

    def error_expect(self, a: str, b: str) -> NoReturn:
        raise Error(self.location, "expected '{}', got '{}'".format(a, b))

    def expect(self, t: typing.Type[T]) -> T:
        current = self.current

        if not isinstance(current, t):
            self.error_expect(t.__name__, type(current).__name__)

        self.advance()

        return current

    def parse_type_atom(self, bound: List[str], free: Dict[str, Type]) -> Type:
        location = self.location

        match self.current:
            case Token_Symbol(value = value):
                self.advance()

                if value in bound:
                    return Type_Var(location, bound.index(value))

                if value not in free:
                    free[value] = fresh_free_type_variable(location)

                return free[value]

            case Token_Type_Real():
                self.advance()
                return Type_Real(location)

            case _:
                self.error_expect("type", type(self.current).__name__)

    def parse_type(self, bound: List[str], free: Dict[str, Type]) -> Type:
        location = self.location

        match self.current:
            case Token_LBracket():
                self.advance()

                lhs: List[Type] = []
                rhs: List[Type] = []

                while True:
                    lhs.append(self.parse_type(bound, free))

                    if isinstance(self.current, (Token_EOF, Token_Minus_Minus)):
                        break

                self.expect(Token_Minus_Minus)

                while True:
                    rhs.append(self.parse_type(bound, free))

                    if isinstance(self.current, (Token_EOF, Token_RBracket)):
                        break

                self.expect(Token_RBracket)

                effect_lhs = stack_from_list(*lhs)
                effect_rhs = stack_from_list(*rhs)

                return Type_Effect(location, effect_lhs, effect_rhs)

            case _:
                return self.parse_type_atom(bound, free)

    def parse_scheme_base(self) -> Scheme:
        match self.current:
            case Token_Forall():
                self.advance()

                bound: List[str] = []

                while True:
                    bound.append(self.expect(Token_Symbol).value)

                    if isinstance(self.current, (Token_EOF, Token_Dot)):
                        break

                self.expect(Token_Dot)

                return Scheme(set(range(len(bound))), self.parse_type(bound, {}))

            case _:
                return Scheme(set(), self.parse_type([], {}))

    def parse_scheme(self) -> Scheme:
        return self.parse_scheme_base()

    def parse_atom(self) -> Tree:
        location = self.location

        match self.current:
            case Token_Bang():
                self.advance()
                return Tree_Let(location, self.expect(Token_Symbol).value)

            case Token_At():
                self.advance()
                return Tree_Def(location, self.expect(Token_Symbol).value, self.parse_scheme())

            case Token_If():
                self.advance()
                return Tree_If(location)

            case Token_While():
                self.advance()
                return Tree_While(location)

            case Token_LBracket():
                self.next_location.append(location)

                self.advance()

                block = self.parse_block()

                self.expect(Token_RBracket)

                return block

            case Token_Plus():
                self.advance()
                return Tree_Add(location)

            case Token_Minus():
                self.advance()
                return Tree_Sub(location)

            case Token_Star():
                self.advance()
                return Tree_Mul(location)

            case Token_Slash():
                self.advance()
                return Tree_Div(location)

            case Token_Real(value=value):
                self.advance()
                return Tree_Real(location, value)

            case Token_Unit():
                self.advance()
                return Tree_Unit(location)

            case Token_Symbol(value=value):
                self.advance()
                return Tree_Symbol(location, value)

            case Token_Print():
                self.advance()
                return Tree_Print(location)

            case Token_Do():
                self.advance()
                return Tree_Do(location)

            case Token_Type():
                self.advance()
                return Tree_Type(location, self.parse_scheme())

            case _:
                self.error_expect("atom", type(self.current).__name__)

    def parse_list(self) -> List[Tree]:
        match self.current:
            case Token_LBrace():
                self.advance()

                body: List[Tree] = []

                while not isinstance(self.current, (Token_EOF, Token_RBrace)):
                    body.append(self.parse_atom())

                self.expect(Token_RBrace)

                return body[::-1]

            case _:
                return [self.parse_atom()]

    def parse_block(self) -> Tree_Block:
        block = Tree_Block(self.location, self.block_id, [])

        self.block_id += 1

        while not isinstance(self.current, (Token_EOF, Token_RBracket)):
            if isinstance(self.current, Token_Import):
                self.advance()

                path = self.expect(Token_String).value

                if "." not in path:
                    path += ".stabl"

                if path in self.importing:
                    raise Error(self.location, f"'{path}' already imported")

                text = try_read(path)

                parser = Parser(text, path, start_block_id=self.block_id, importing=self.importing)

                block.body.extend(parser.parse().body)

                self.block_id = parser.block_id
            else:
                block.body.extend(self.parse_list())

        return block

    def parse(self) -> Tree_Block:
        return self.parse_block()



class Scope(Generic[T]):
    def __init__(self, default_factory: Callable[[], T], parent: Optional['Scope[T]'] = None):
        self.parent = parent

        self.lets: T = default_factory()
        self.defs: T = default_factory()


## Infer


def validate_effect(effect: Type_Effect):
    for tv in effect.rhs.free_type_variables_exclude_effect():
        if tv not in reachable_variables(effect):
            s1, s2 = align_types(effect.lhs, effect.rhs)

            msg = f"""
  ill shaped effect, '{effect}'
  no correlation between
         left-hand side '{s1}'
    and right-hand side '{s2}'"""
            raise Error(effect.location, msg)


def reachable_variables(effect: Type_Effect) -> Set[int]:
    def collect_effects(t: Type) -> List[Type_Effect]:
        if isinstance(t, Type_Stack):
            effects = collect_effects(t.tail)
            if isinstance(t.head, Type_Effect):
                effects.append(t.head)
            return effects
        return []

    known = effect.lhs.free_type_variables_exclude_effect()
    lhs_effects = collect_effects(effect.lhs)

    changed = True
    while changed:
        changed = False
        for e in lhs_effects:
            inputs = e.lhs.free_type_variables()
            if inputs <= known:
                outputs = e.rhs.free_type_variables()
                new = outputs - known
                if new:
                    known |= new
                    changed = True

    return known


def infer(block: Tree_Block, parent: Optional[Scope[Dict[str, Scheme]]] = None) -> Tuple[Type_Effect, Substitution]:
    sv = fresh_free_type_variable(block.location)

    result = Type_Effect(block.location, sv, sv)

    scope: Scope[Dict[str, Scheme]] = Scope(dict, parent)

    s: Substitution = {}

    def apply(r: Substitution):
        nonlocal s, result

        s = compose_substitution(s, r)

        result = result.apply(s)

        current: Scope[Dict[str, Scheme]] = scope

        while True:
            current.lets = {k: v.apply(s) for k, v in current.lets.items()}
            current.defs = {k: v.apply(s) for k, v in current.defs.items()}

            if current.parent is None:
                break

            current = current.parent

    for tree in block.body:
        l = tree.location

        scheme: Scheme

        match tree:
            case Tree_Real():
                scheme = Scheme({0},
                    Type_Effect(l,
                        stack_from_list(Type_Var(l, 0)),
                        stack_from_list(Type_Var(l, 0), Type_Real(l)),
                    )
                )

            case Tree_Add() | Tree_Sub() | Tree_Mul() | Tree_Div():
                scheme = Scheme({0},
                    Type_Effect(l,
                        stack_from_list(Type_Var(l, 0), Type_Real(l), Type_Real(l)),
                        stack_from_list(Type_Var(l, 0), Type_Real(l)),
                    )
                )

            case Tree_Let(name = name):
                tv = fresh_free_type_variable(l)

                scheme = Scheme({0},
                    Type_Effect(l,
                        stack_from_list(Type_Var(l, 0), tv),
                        stack_from_list(Type_Var(l, 0)),
                    )
                )

                scope.lets[name] = Scheme(set(), tv)

            case Tree_Def(name = name, t = t):
                u = t.instantiate()

                scheme = Scheme({0},
                    Type_Effect(l,
                        stack_from_list(Type_Var(l, 0), u),
                        stack_from_list(Type_Var(l, 0)),
                    )
                )

                scope.defs[name] = t

            case Tree_Symbol(value = value):
                current: Scope[Dict[str, Scheme]] = scope

                while True:
                    if value in current.lets:
                        t = current.lets[value]

                        scheme = Scheme({0},
                            Type_Effect(l,
                                stack_from_list(Type_Var(l, 0)),
                                stack_from_list(Type_Var(l, 0), t.instantiate()),
                            )
                        )

                        break

                    if value in current.defs:
                        scheme = current.defs[value]
                        break

                    if current.parent is None:
                        raise Error(tree.location, "undefined '{}'".format(value))

                    current = current.parent

            case Tree_Block():
                typ, sub = infer(tree, scope)

                apply(sub)

                scheme = Scheme({0},
                    Type_Effect(l,
                        stack_from_list(Type_Var(l, 0)),
                        stack_from_list(Type_Var(l, 0), typ),
                    )
                )

            case Tree_If():
                s1 = Type_Var(l, 0)
                s2 = Type_Var(l, 1)
                q = Type_Effect(l, s1, s2)

                scheme = Scheme({0, 1},
                    Type_Effect(l,
                        stack_from_list(s1, Type_Real(l), q, q),
                        stack_from_list(s2),
                    )
                )

            case Tree_While():
                s1 = Type_Var(l, 0)

                q_cond = Type_Effect(l, stack_from_list(s1), stack_from_list(s1, Type_Real(l)))
                q_body = Type_Effect(l, stack_from_list(s1), stack_from_list(s1))

                scheme = Scheme({0},
                    Type_Effect(l,
                        stack_from_list(s1, q_cond, q_body),
                        stack_from_list(s1),
                    )
                )

            case Tree_Print():
                scheme = Scheme({0, 1},
                    Type_Effect(l,
                        stack_from_list(Type_Var(l, 0), Type_Var(l, 1)),
                        stack_from_list(Type_Var(l, 0)),
                    )
                )

            case _:
                print(f"\033[95mWARNING\033[0m: node '{type(tree).__name__}' is not handled yet") # ]]
                continue

        effect = scheme.instantiate()

        if not isinstance(effect, Type_Effect):
            raise Error(l, "cannot apply non-effect type '{}' to stack".format(effect))

        result, result_s = compose_effect(result, effect)

        apply(result_s)

    validate_effect(result)

    return result, s


## Compiler

def tree_captures(block: Tree_Block) -> Tuple[Set[str], Set[str]]:
    mentioned: Set[str] = set()

    lets: Set[str] = set()
    defs: Set[str] = set()

    for t in block.body:
        match t:
            case Tree_Let(name = name):
                lets.add(name)

            case Tree_Def(name = name):
                defs.add(name)

            case Tree_Symbol(value = value):
                mentioned.add(value)

            case Tree_Block():
                block_lets, block_defs = tree_captures(t)
                mentioned.update(block_lets | block_defs)

    return mentioned - lets, mentioned - defs


impossible_count = 0

def tree_rename_impossible(block: Tree_Block, lets0 = None, defs0 = None):
    global impossible_count

    lets = lets0 or set()
    defs = defs0 or set()

    new_body: List[Tree] = []

    for s in block.body:
        location = s.location

        match s:
            case Tree_Let(name = name):
                lets.add(name)

                new_body.append(Tree_Let(location, f"{name} {impossible_count}"))

            case Tree_Def(name = name, t = t):
                defs.add(name)

                new_body.append(Tree_Def(location, f"{name} {impossible_count}", t))

            case Tree_Symbol(value = value):
                if value in lets:
                    new_body.append(Tree_Symbol(location, f"{value} {impossible_count}"))
                elif value in defs:
                    new_body.append(Tree_Symbol(location, f"{value} {impossible_count}"))
                else:
                    new_body.append(s)

            case Tree_Block():
                new_body.append(tree_rename_impossible(s, lets, defs))

            case _:
                new_body.append(s)

    captured_lets = { f"{x} {impossible_count}" if x in lets else x for x in block.captured_lets }
    captured_defs = { f"{x} {impossible_count}" if x in defs else x for x in block.captured_defs }

    return Tree_Block(block.location, block.id, new_body, block.name, captured_lets, captured_defs)



def tree_inline(block: Tree_Block, parent_defs: dict | None = None):
    global impossible_count

    local_defs = parent_defs or {}
    prev = None
    new_body: List[Tree] = []

    for s in block.body:
        if isinstance(s, Tree_Block):
            tree_inline(s, parent_defs=local_defs.copy())

        match s:
            case Tree_Def(name = name):
                if isinstance(prev, Tree_Block):
                    local_defs[name] = prev
                    new_body.pop(-1)
                else:
                    new_body.append(s)

            case Tree_Symbol(value=value):
                if value in local_defs:
                    for t in tree_rename_impossible(local_defs[value]).body:
                        new_body.append(t)

                else:
                    new_body.append(s)

                impossible_count += 1

            case _:
                new_body.append(s)

        prev = s

    block.captured_defs.difference_update(local_defs.keys())
    block.body[:] = new_body

def tree_resolve_names(block: Tree_Block, parent: Optional[Scope] = None):
    scope = Scope[Set[str]](set, parent)

    used_lets = set()
    used_defs = set()

    for s in block.body:
        location = s.location

        match s:
            case Tree_Let(name = name):
                scope.lets.add(name)

            case Tree_Def(name = name):
                scope.defs.add(name)

            case Tree_Symbol(value = value):
                current: Scope = scope

                while True:
                    if value in current.lets:
                        used_lets.add(value)
                        break

                    elif value in current.defs:
                        used_defs.add(value)
                        break

                    if current.parent is None:
                        raise Error(location, "undefined `{}' (tree_resolve_names)".format(value))

                    current = current.parent

            case Tree_Block():
                tree_resolve_names(s, scope)

                used_lets.update(s.captured_lets)
                used_defs.update(s.captured_defs)

    captured_lets = used_lets - scope.lets
    captured_defs = used_defs - scope.defs

    block.captured_lets.update(captured_lets)
    block.captured_defs.update(captured_defs)


def flatten(block: Tree_Block) -> List[Tree_Block]:
    stack: List[Tree_Block] = [block]
    drain: List[Tree_Block] = []

    while stack:
        block = stack.pop()

        for tree in block.body:
            if isinstance(tree, Tree_Block):
                stack.insert(0, tree)

        drain.insert(0, block)

    return drain


def tree_need_sstack(block: Tree_Block) -> bool:
    defs = set().union(block.captured_defs)

    for tree in block.body:
        match tree:
            case Tree_Def(name = name):
                defs.add(name)

            case Tree_While():
                return True

            case Tree_Symbol(value = value):
                if value in defs:
                    return True

            case Tree_Do():
                return True

    return False


def tree_gc_weight(block: Tree_Block, parent: Optional[Scope[Set[str]]] = None) -> float:
    n = 0

    scope: Scope[Set[str]] = Scope(set, parent)

    scope.lets = set().union(block.captured_lets)
    scope.defs = set().union(block.captured_defs)

    for tree in block.body:
        match tree:
            case Tree_Let(name = name):
                scope.lets.add(name)

            case Tree_Def(name = name):
                scope.defs.add(name)

            case Tree_Symbol(value = value):
                current: Scope = scope

                while True:
                    if value in current.lets:
                        n += 0
                        break

                    elif value in current.defs:
                        n += 1
                        break

                    if current.parent is None:
                        raise Error(tree.location, "undefined `{}'".format(value))

                    current = current.parent

            case Tree_Block():
                if tree.is_closure:
                    n += 1
                else:
                    n += 1

            case Tree_Print():
                n += 2

            case _:
                n += 0

    l = len(block.body)

    if l != 0:
        return n / l
    else:
        return 0


def tree_need_gc(block: Tree_Block) -> bool:
    return tree_gc_weight(block) >= 0.5


class Compiler:
    def __init__(self, blocks: List[Tree_Block], generate_debug=False) -> None:
        self.blocks = blocks

        self.lets: Dict[str, int] = {}
        self.defs: Dict[str, int] = {}

        self.done: Set[int] = set()

        self.slot_id = 0

        self.generate_debug = generate_debug


    @staticmethod
    def block_name(id: int) -> str:
        return "block_{}".format(id)


    @staticmethod
    def slot_name(id: int) -> str:
        return "slot_{}".format(id)


    def compile_block(self, block: Tree_Block):
        need_sstack = tree_need_sstack(block)
        need_gc     = tree_need_gc(block)

        self.lets = {}
        self.defs = {}

        self.slot_id = 0

        readables = []

        if block.name:
            readables.append("name={}".format(block.name))
        if block.is_closure:
            readables.append("+is_closure")
        if need_sstack:
            readables.append("+need_sstack")
        if need_gc:
            readables.append("+need_gc({:.2f})".format(tree_gc_weight(block)))

        if readables:
            print("// {}".format(" ".join(readables)))

        if block.is_closure:
            print(f"void\n{self.block_name(block.id)} (struct value **env)\n{{") # }}
        else:
            if block.id != 0:
                print(f"static inline ", end="")
            print(f"void\n{self.block_name(block.id)} (void)\n{{") # }}

        if self.generate_debug:
            if block.name:
                name = "{}".format(block.name)
            else:
                name = "{} \\\\ {}:{}:{}".format(self.block_name(block.id), *block.location)

            print(f"  cstack_push (\"{name}\");")

            print()


        if block.is_closure:
            capture_i = 0

            for capture in sorted(list(block.captured_lets)):
                if capture in self.lets or capture in self.lets:
                    raise Error(block.location, "redefined `{}'".format(capture));

                self.lets[capture] = self.slot_id

                print(f"  struct value *{self.slot_name(self.slot_id)} = env[{capture_i}]; // !{capture}")

                if need_sstack:
                    print ()

                    print(f"  sstack_push ({self.slot_name(self.slot_id)});")

                    print ()

                self.slot_id += 1

                capture_i += 1

            for capture in sorted(list(block.captured_defs)):
                if capture in self.lets or capture in self.defs:
                    raise Error(block.location, "redefined `{}'".format(capture));

                self.defs[capture] = self.slot_id

                print(f"  struct value *{self.slot_name(self.slot_id)} = env[{capture_i}]; // @{capture}")

                if need_sstack:
                    print ()

                    print(f"  sstack_push ({self.slot_name(self.slot_id)});")

                    print ()

                self.slot_id += 1

                capture_i += 1

        for tree in block.body:
            location = tree.location

            location_str = "{}:{}:{}".format(*location)

            if self.generate_debug:
                print("  location = \"{}\";".format(location_str))
                print()

            match tree:
                case Tree_Add():
                    print("  {") # }
                    print("    const f64 b = value_unbox_f (vstack_pop ());")
                    print("    const f64 a = value_unbox_f (vstack_pop ());")
                    print("    vstack_push (value_box_f (a + b));")
                    if self.generate_debug:
                        print("    vstack_append_trace (value_trace_create (\"{}\", \"+\"));".format(location_str))
                    print("  }")

                case Tree_Sub():
                    print("  {") # }
                    print("    const f64 b = value_unbox_f (vstack_pop ());")
                    print("    const f64 a = value_unbox_f (vstack_pop ());")
                    print("    vstack_push (value_box_f (a - b));")
                    if self.generate_debug:
                        print("    vstack_append_trace (value_trace_create (\"{}\", \"-\"));".format(location_str))
                    print("  }")

                case Tree_Mul():
                    print("  {") # }
                    print("    const f64 b = value_unbox_f (vstack_pop ());")
                    print("    const f64 a = value_unbox_f (vstack_pop ());")
                    print("    vstack_push (value_box_f (a * b));")
                    if self.generate_debug:
                        print("    vstack_append_trace (value_trace_create (\"{}\", \"*\"));".format(location_str))
                    print("  }")

                case Tree_Div():
                    print("  {") # }
                    print("    const f64 b = value_unbox_f (vstack_pop ());")
                    print("    const f64 a = value_unbox_f (vstack_pop ());")
                    print("    vstack_push (value_box_f (a / b));")
                    if self.generate_debug:
                        print("    vstack_append_trace (value_trace_create (\"{}\", \"/\"));".format(location_str))
                    print("  }")

                case Tree_Let(name = name):
                    if name in self.lets or name in self.defs:
                        raise Error(location, "redefined `{}'".format(name));

                    self.lets[name] = self.slot_id

                    print(f"  struct value *{self.slot_name(self.slot_id)} = vstack_pop (); // !{name}")

                    if self.generate_debug:
                        print("  value_append_trace ({}, value_trace_create (\"{}\", \"!{}\"));".format(self.slot_name(self.slot_id), location_str, name))

                    if need_sstack:
                        print ()

                        print(f"  sstack_push ({self.slot_name(self.slot_id)});")

                    self.slot_id += 1

                case Tree_Def(name = name):
                    if name in self.lets or name in self.defs:
                        raise Error(location, "redefined `{}'".format(name));

                    self.defs[name] = self.slot_id

                    print(f"  struct value *{self.slot_name(self.slot_id)} = vstack_pop (); // @{name}")

                    if self.generate_debug:
                        print("  value_append_trace ({}, value_trace_create (\"{}\", \"@{}\"));".format(self.slot_name(self.slot_id), location_str, name))

                    if need_sstack:
                        print ()

                        print(f"  sstack_push ({self.slot_name(self.slot_id)});")

                    self.slot_id += 1

                case Tree_If():
                    print( "  {") # }
                    print( "    struct value *b = vstack_pop ();")
                    print( "    struct value *a = vstack_pop ();")
                    print()
                    print( "    if (value_bool (vstack_pop ()))")
                    print( "      {") # }
                    print(f"        value_execute (a);")
                    print( "      }")
                    print( "    else")
                    print( "      {") # }
                    print(f"        value_execute (b);")
                    print( "      }")
                    print( "  }")

                case Tree_While():
                    print("  {") # }
                    print("    struct value *body = vstack_pop ();")
                    print("    struct value *cond = vstack_pop ();")
                    print()
                    print("    sstack_push (body);")
                    print("    sstack_push (cond);")
                    print()
                    print("    while (1)")
                    print("      {") # }
                    print("        value_execute (cond);")
                    print()
                    print("        if (!value_bool (vstack_pop ()))")
                    print("          break;")
                    print()
                    print("        value_execute (body);")
                    print("      }") # }
                    print()
                    print("    sstack_pop ();")
                    print("    sstack_pop ();")
                    print("  }")

                case Tree_Block(id = id):
                    if tree.is_closure:
                        compound = []

                        capture_i = 0

                        for x in sorted(list(tree.captured_lets)):
                            compound.append(f"{self.slot_name(self.lets[x])}")
                            capture_i += 1

                        for x in sorted(list(tree.captured_defs)):
                            compound.append(f"{self.slot_name(self.defs[x])}")
                            capture_i += 1

                        literal = f"(struct value *[]) {{ {', '.join(compound)} }}"
                        print(f"  vstack_push (value_box_c ({literal}, {capture_i}, {self.block_name(tree.id)}));")

                    else:
                        print(f"  vstack_push (value_box_b ({self.block_name(tree.id)}));")

                case Tree_Real(value = value):
                    print(f"  vstack_push (value_box_f ({float(value)}));")

                    if self.generate_debug:
                        print("  vstack_append_trace (value_trace_create (\"{}\", \"{}\"));".format(location_str, value))

                case Tree_Unit():
                    print(f"  vstack_push (value_box_u ());")

                    if self.generate_debug:
                        print("  vstack_append_trace (value_trace_create (\"{}\", \"{}\"));".format(location_str, "()"))

                case Tree_Symbol(value = value):
                    if value in self.lets:
                        print(f"  vstack_push ({self.slot_name (self.lets[value])});")
                        if self.generate_debug:
                            print("  vstack_append_trace (value_trace_create (\"{}\", \"{}\"));".format(location_str, value))

                    elif value in self.defs:
                        print(f"  value_execute ({self.slot_name (self.defs[value])});")

                    else:
                        raise Error(location, "undefined `{}'".format(value))

                case Tree_Print():
                    print(f"  value_printn (vstack_pop ());")

            print()

        # Cleanup block

        for i, id in enumerate(range(self.slot_id)):
            if i > 0:
                print()

            if need_sstack:
                print(f"  sstack_pop ();")
            else:
                print(f"  UNUSED ({self.slot_name(id)});")

        if need_gc:
            print()

            print("  gc_collect_try ();")

        if self.generate_debug:
            print()

            print("  cstack_pop ();")

        print(f"}}")

    def compile(self):
        if self.generate_debug:
            print("#define DEBUG\n")

        print("#include \"runtime/runtime-main.c\"\n\n")

        for i, block in enumerate(self.blocks):
            if block.id in self.done:
                continue

            if i > 0:
                print()

            self.compile_block(block)

            print()

            self.done.add(block.id)
            # print()




## Entry

def try_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        raise Error(None, str(e))


CLEAR_LINE = "\033[2K\r" #]


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <file>", file=sys.stderr)
        exit(1)

    PATH = sys.argv[1]

    BASE   = PATH[:-len(".stabl")]
    PATH_C = BASE + ".c"

    try:
        print(f"Parsing...", end="", file=sys.stderr, flush=True)

        tree = Parser(try_read(PATH), PATH).parse()

        print(f"{CLEAR_LINE}Checking...", end="", file=sys.stderr, flush=True)

        t, _ = infer(tree)

        if not isinstance(t.lhs, Type_Var) or not isinstance(t.rhs, Type_Var):
            raise Error(tree.location, "\n  main program must not consume or produce anything;\n  but it has type '{}'.".format(t))

        print(f"{CLEAR_LINE}Compiling...", end="", file=sys.stderr, flush=True)

        tree_resolve_names(tree)

        # tree_print(tree)

        assert isinstance(tree, Tree_Block)

        blocks = flatten(tree)

        with open(PATH_C, "w") as f:
            with contextlib.redirect_stdout(f):
                Compiler(blocks, False).compile()

        result = subprocess.run(
            ["gcc", "-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-I.", f"-o{BASE}", PATH_C],
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise Error(None, "gcc failed:\n" + result.stderr)

        print(f"{CLEAR_LINE}Done.", file=sys.stderr)

    except Error as e:
        print(file=sys.stderr)
        if e.location:
            print("{}:{}:{}: ".format(*e.location), file=sys.stderr, end="")

        print("\033[1;31merror\033[0m: {}".format(e.message), file=sys.stderr) # ]]
        exit(1)


if __name__ == "__main__":
    main()

