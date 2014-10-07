"""Fermionic quantum operators."""

from warnings import warn

from sympy.core.compatibility import u
from sympy import Add, Mul, Pow, Integer, exp, sqrt, conjugate
from sympy.physics.quantum import Operator, Commutator, AntiCommutator, Dagger
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta


__all__ = [
    'FermionOp',
    'FermionFockKet',
    'FermionFockBra',
    'MultiFermionOp'
]


class FermionOp(Operator):
    """A fermionic operator that satisfies {c, Dagger(c)} == 1.

    Parameters
    ==========

    name : str
        A string that labels the fermionic mode.

    annihilation : bool
        A bool that indicates if the fermionic operator is an annihilation
        (True, default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, AntiCommutator
    >>> from sympy.physics.quantum.fermion import FermionOp
    >>> c = FermionOp("c")
    >>> AntiCommutator(c, Dagger(c)).doit()
    1
    """
    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        return ("c", True)

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)

        if len(args) == 1:
            args = (args[0], Integer(1))

        if len(args) == 2:
            args = (args[0], Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_commutator_FermionOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # [c, d] = 0
            return Integer(0)

        return None

    def _eval_anticommutator_FermionOp(self, other, **hints):
        if self.name == other.name:
            # {a^\dagger, a} = 1
            if not self.is_annihilation and other.is_annihilation:
                return Integer(1)

        elif 'independent' in hints and hints['independent']:
            # {c, d} = 2 * c * d, because [c, d] = 0 for independent operators
            return 2 * self * other

        return None

    def _eval_anticommutator_BosonOp(self, other, **hints):
        # because fermions and bosons commute
        return 2 * self * other

    def _eval_commutator_BosonOp(self, other, **hints):
        return Integer(0)

    def _eval_adjoint(self):
        return FermionOp(str(self.name), not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)
        else:
            return r'{{%s}^\dag}' % str(self.name)

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)
        else:
            return r'Dagger(%s)' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm(u('\u2020'))
            
class MultiFermionOp(Operator):
    """Fermionic operators that satisfy the commutation relations:
    for discrete label for modes:
        {a(k1), Dagger(a(k2))} == KroneckerDelta(k1, k2). 
    
    for continuous label for modes:
        {a(k1), Dagger(a(k2))} == DiracDelta(k1 - k2).
        
    and in both cases:
        {a(k1), a(k2)} == {Dagger(a(k1)), Dagger(a(k2))} == 0.


    Parameters
    ==========

    name : str
        A string that labels the bosonic mode.

    mode: Symbol
        A symbol that denotes the mode label.

    normalization : ['discrete', 'continuous']
        'discrete' for KroneckerDelta function,
        'continuous' for DiracDelta function. 
        should be specified in any case.

    annihilation : bool
        A bool that indicates if the bosonic operator is an annihilation (True,
        default value) or creation operator (False)
    
        
    Examples
    ========
    
    >>> from sympy.physics.quantum import Dagger, Commutator
    >>> from sympy.physics.quantum.fermion import MultiFermionOp
    >>> w1, w2 = symbols("w1, w2")
    >>> a1 = MultiFermionOp("a", w1, 'discrete')
    >>> a2 = MultiFermionOp("a", w2, 'discrete')
    >>> Commutator(a1, Dagger(a2)).doit()
    KroneckerDelta(w1, w2)
    >>> Commutator(a1, a2).doit()
    0
    >>> Commutator(Dagger(a1), Dagger(a2)).doit()
    0
    >>> b1 = MultiFermionOp("b", w1, 'continuous')
    >>> b2 = MultiFermionOp("b", w2, 'continuous')
    >>> AntiCommutator(b1, Dagger(b2)).doit()
    DiracDelta(w1 - w2)
    >>> AntiCommutator(b1, b2).doit()
    0
    >>> AntiCommutator(Dagger(b1), Dagger(b2)).doit()
    0
    
    """    
    
    @property
    def free_symbols(self):
        return self.args[1].free_symbols
    
    @property
    def name(self):
        return self.args[0]
    
    @property
    def mode(self):
        return self.args[1]
        
    @property
    def normalization_type(self):
        return str(self.args[3])

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        return ("a", Symbol("\omega"), "discrete", True)

    def __new__(cls, *args, **hints):
        if not len(args) in [3, 4]:
            raise ValueError('3 or 4 parameters expected, got %s' % args)

        if str(args[2]) not in ['discrete', 'continuous']:
            print("discrete or continuous: %s" % args[2])
            raise ValueError('The third argument should be "discrete" or "continuous", got %s' % args)
        
        if len(args) == 3:
            args = (args[0], args[1], str(args[2]), Integer(1))

        if len(args) == 4:
            args = (args[0], args[1], str(args[2]), Integer(args[3]))

        return Operator.__new__(cls, *args)


#########
    def _eval_commutator_FermionOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # [c, d] = 0
            return Integer(0)

        return None
    

    def _eval_anticommutator_FermionOp(self, other, **hints):
        if self.name == other.name:
            # {a^\dagger, a} = 1
            if not self.is_annihilation and other.is_annihilation:
                return Integer(1)

        elif 'independent' in hints and hints['independent']:
            # {c, d} = 2 * c * d, because [c, d] = 0 for independent operators
            return 2 * self * other

        return None

    def _eval_anticommutator_BosonOp(self, other, **hints):
        # because fermions and bosons commute
        return 2 * self * other

    def _eval_commutator_BosonOp(self, other, **hints):
        return Integer(0)

    def _eval_adjoint(self):
        return FermionOp(str(self.name), not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)
        else:
            return r'{{%s}^\dag}' % str(self.name)

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)
        else:
            return r'Dagger(%s)' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm(u('\u2020'))


class FermionFockKet(Ket):
    """Fock state ket for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if n not in [0, 1]:
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return FermionFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_FermionFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_operator_FermionOp(self, op, **options):
        if op.is_annihilation:
            if self.n == 1:
                return FermionFockKet(0)
            else:
                return Integer(0)
        else:
            if self.n == 0:
                return FermionFockKet(1)
            else:
                return Integer(0)


class FermionFockBra(Bra):
    """Fock state bra for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if n not in [0, 1]:
            raise ValueError("n must be 0 or 1")
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return FermionFockKet
