"""Bosonic quantum operators."""

from sympy.core.compatibility import u
from sympy import Mul, Integer, exp, sqrt, conjugate, DiracDelta, Symbol
from sympy.physics.quantum import Operator, Commutator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator
from sympy.functions.special.tensor_functions import KroneckerDelta


__all__ = [
    'BosonOp',
    'MultiBosonOp',
    'BosonFockKet',
    'BosonFockBra',
    'BosonCoherentKet',
    'BosonCoherentBra',
    'MultiBosonFockKet',
    'MultiBosonFockBra',
    'BosonVacuumKet',
    'BosonVacuumBra'
]


class BosonOp(Operator):
    """A bosonic operator that satisfies [a, Dagger(a)] == 1.

    Parameters
    ==========

    name : str
        A string that labels the bosonic mode.

    annihilation : bool
        A bool that indicates if the bosonic operator is an annihilation (True,
        default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, Commutator
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> a = BosonOp("a")
    >>> Commutator(a, Dagger(a)).doit()
    1
    """

    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @property
    def free_symbols(self):
        return set([])

    @classmethod
    def default_args(self):
        return ("a", True)

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % str(args))

        if len(args) == 1:
            args = (args[0], Integer(1))

        if len(args) == 2:
            args = (args[0], Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_commutator_BosonOp(self, other, **hints):
        if self.name == other.name:
            # [a^\dagger, a] = -1
            if not self.is_annihilation and other.is_annihilation:
                return Integer(-1)

        elif 'independent' in hints and hints['independent']:
            # [a, b] = 0
            return Integer(0)

        return None

    def _eval_commutator_FermionOp(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_BosonOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # {a, b} = 2 * a * b, because [a, b] = 0
            return 2 * self * other

        return None

    def _eval_adjoint(self):
        return BosonOp(str(self.name), not self.is_annihilation)

    def __mul__(self, other):

        if other == IdentityOperator(2):
            return self

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

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


class MultiBosonOp(BosonOp):
    """Bosonic operators that satisfy the commutation relations:
    for discrete label for modes:
        [a(k1), Dagger(a(k2))] == KroneckerDelta(k1, k2). 
    
    for continuous label for modes:
        [a(k1), Dagger(a(k2))] == DiracDelta(k1 - k2).
        
    and in both cases:
        [a(k1), a(k2)] == [Dagger(a(k1)), Dagger(a(k2))] == 0.


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
    >>> from sympy.physics.quantum.boson import MultiBosonOp
    >>> w1, w2 = symbols("w1, w2")
    >>> a1 = MultiBosonOp("a", w1, 'discrete')
    >>> a2 = MultiBosonOp("a", w2, 'discrete')
    >>> Commutator(a1, Dagger(a2)).doit()
    KroneckerDelta(w1, w2)
    >>> Commutator(a1, a2).doit()
    0
    >>> Commutator(Dagger(a1), Dagger(a2)).doit()
    0
    >>> b1 = MultiBosonOp("b", w1, 'continuous')
    >>> b2 = MultiBosonOp("b", w2, 'continuous')
    >>> Commutator(b1, Dagger(b2)).doit()
    DiracDelta(w1 - w2)
    >>> Commutator(b1, b2).doit()
    0
    >>> Commutator(Dagger(b1), Dagger(b2)).doit()
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
        return str(self.args[2])

    @property
    def is_annihilation(self):
        return bool(self.args[3])

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

    def _eval_commutator_BosonOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # [a, b] = 0
            return Integer(0)

    def _eval_commutator_MultiBosonOp(self, other, **hints):
        if (self.name == other.name and 
            self.normalization_type == other.normalization_type):
            if not self.is_annihilation and other.is_annihilation:
                if self.normalization_type == 'discrete':
                    return - KroneckerDelta(self.mode, other.mode)
                elif self.normalization_type == 'continuous':
                    return - DiracDelta(self.mode - other.mode)
            elif not self.is_annihilation and not other.is_annihilation:
                return Integer(0)
            elif self.is_annihilation and other.is_annihilation:
                return Integer(0)
                
        elif 'independent' in hints and hints['independent']:
            # [a, b] = 0
            return Integer(0)

        return None

    def _eval_commutator_FermionOp(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_BosonOp(self, other, **hints):
        return 2 * self * other

    def _eval_anticommutator_MultiBosonOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # {a, b} = 2 * a * b, because [a, b] = 0
            return 2 * self * other
        else:
            return 2 * self * other - Commutator(self, other)

        return None

    def _eval_adjoint(self):
        return MultiBosonOp(self.name, self.mode, self.normalization_type, not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s_{%s}}' % (str(self.name), str(self.mode))
        else:
            return r'{{%s_{%s}}^\dag}' % (str(self.name), str(self.mode))

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s(%s)' % (str(self.name), str(self.mode))
        else:
            return r'Dagger(%s(%s))' % (str(self.name), str(self.mode))

    def _print_contents_pretty(self, printer, *args):
        # TODO
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm(u('\u2020'))


class BosonFockKet(Ket):
    """Fock state ket for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.
        

    """

    def __new__(cls, n):
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    def _eval_innerproduct_BosonFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_operator_BosonOp(self, op, **options):
        if op.is_annihilation:
            return sqrt(self.n) * BosonFockKet(self.n - 1)
        else:
            return sqrt(self.n + 1) * BosonFockKet(self.n + 1)

class BosonFockBra(Bra):
    """Fock state bra for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonFockKet

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

class MultiBosonFockKet(Ket):
    """Fock state ket for a multimode-bosonic mode(KroneckerDelta normalization). 

    Parameters
    ==========

    n : Number
        The Fock state number.

    mode : Symbol
        A symbol that denotes the mode label.  

    """

    def __new__(cls, n, mode):
        return Ket.__new__(cls, n, mode)

    @property
    def n(self):
        return self.label[0]

    @property    
    def mode(self):
        return self.label[1]
        
    @classmethod
    def default_args(self):
        return (Integer(0), Symbol("k"))
        
    @classmethod
    def dual_class(self):
        return MultiBosonFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    def _eval_innerproduct_MultiBosonFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n) * KroneckerDelta(self.mode, bra.mode)

    def _apply_operator_MultiBosonOp(self, op, **options):
        if op.normalization_type == 'discrete':
            if op.mode != self.mode:
                return None
            if op.is_annihilation:
                return sqrt(self.n) * MultiBosonFockKet(self.n - 1, op.mode)
            else:
                return sqrt(self.n + 1) * MultiBosonFockKet(self.n + 1, op.mode)
        else:
            return None

    def _latex(self, printer, *args):
        return r'{\left| {%s} \right\rangle}{_{%s}}' % (str(self.n), str(self.mode))
        
class MultiBosonFockBra(Bra):
    """Fock state bra for a multimode-bosonic mode(KroneckerDelta normalization). 

    Parameters
    ==========

    n : Number
        The Fock state number.

    mode : Symbol
        A symbol that denotes the mode label.  

    """

    def __new__(cls, n, mode):
        return Ket.__new__(cls, n, mode)

    @property
    def n(self):
        return self.label[0]

    @property    
    def mode(self):
        return self.label[1]
    
    @classmethod
    def dual_class(self):
        return MultiBosonFockKet

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()
    
    def _latex(self, printer, *args):
        return r'{_{%s}}{\left\langle {%s} \right|}' % (str(self.mode), str(self.n))

class BosonCoherentKet(Ket):
    """Coherent state ket for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Ket.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_BosonCoherentBra(self, bra, **hints):
        if self.alpha == bra.alpha:
            return Integer(1)
        else:
            return exp(-(abs(self.alpha)**2 + abs(bra.alpha)**2 - 2 * conjugate(bra.alpha) * self.alpha)/2)

    def _apply_operator_BosonOp(self, op, **options):
        if op.is_annihilation:
            return self.alpha * self
        else:
            return None


class BosonCoherentBra(Bra):
    """Coherent state bra for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Bra.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentKet

    def _apply_operator_BosonOp(self, op, **options):
        if not op.is_annihilation:
            return self.alpha * self
        else:
            return None
            
            
class BosonVacuumKet(Ket):
    """
    Ket representing the Vacuum State. |vac>
    returns zero if an annihilation operator is applied on the left.
    
    """
    def __new__(cls):
        return Ket.__new__(cls)

    @classmethod
    def dual_class(self):
        return BosonVacuumBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    def _eval_innerproduct_BosonVacuumBra(self, bra, **hints):
        return Integer(1)
    
    def _apply_operator_BosonOp(self, op, **options):
        if op.is_annihilation:
            return Integer(0)
        else:
            return None
            
    def _apply_operator_MultiBosonOp(self, op, **options):
        if op.is_annihilation:
            return Integer(0)
        else:
            return None
            
    def _latex(self, printer, *args):
        return r'{\left| \mathrm{vac} \right\rangle}'

class BosonVacuumBra(Bra):
    """Bra representing the Vacuum State. <vac|
    returns zero if a creation operator is applied on the right.

    """

    def __new__(cls):
        return Bra.__new__(cls)

    @classmethod
    def dual_class(self):
        return BosonVacuumKet

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    def _apply_operator_BosonOp(self, op, **options):
        if not op.is_annihilation:
            return Integer(0)
        else:
            return None

    def _apply_operator_MultiBosonOp(self, op, **options):
        if not op.is_annihilation:
            return Integer(0)
        else:
            return None
            
    def _latex(self, printer, *args):
        return r'{\left< \mathrm{vac} \right|}'