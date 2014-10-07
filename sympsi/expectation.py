"""Expectation values and other statistical measures for operators: ``<A>"""

from __future__ import print_function, division

from sympy import Expr, Add, Mul, Integer, Symbol, Integral
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qapply import qapply


__all__ = [
    'Expectation',
    'Covariance'
]

#-----------------------------------------------------------------------------
# Expectation
#-----------------------------------------------------------------------------

class Expectation(Expr):
    """
    Expectation Value of an operator, expressed in terms of bracket <A>.
    
    If the second argument, 'is_normal_order' is 'True',
    the normal ordering notation (<: :>) is attached.

    doit() returns the normally ordered operator inside the bracket.

    Parameters
    ==========
    
    A : Expr
        The argument of the expectation value <A>

    is_normal_order : bool
        A bool that indicates if the operator inside the Expectation
        value bracket should be normally ordered (True) or left
        untouched (False, default value)
    
    Examples
    ========

    >>> a = BosonOp("a")
    >>> Expectation(a * Dagger(a))
    <a a†>
    >>> Expectation(a * Dagger(a), True)
    <:a a†:>
    >>> Expectation(a * Dagger(a), True).doit()
    <a† a>
    
    """
    is_commutative = True

    @property    
    def expression(self):
        return self.args[0]

    @property    
    def is_normal_order(self):
        return bool(self.args[1])
    
    @classmethod
    def default_args(self):
        return (Symbol("A"), False)
        
    def __new__(cls, *args):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % str(args))
        if len(args) == 1:
            args = (args[0], Integer(0))
        if len(args) == 2:
            args = (args[0], Integer(args[1]))
        return Expr.__new__(cls, *args)
    
    def _eval_expand_expectation(self, **hints):
        A = self.args[0]
        if isinstance(A, Add):
            # <A + B> = <A> + <B>
            return Add(*(Expectation(a, self.is_normal_order).expand(expectation=True) for a in A.args))

        if isinstance(A, Mul):
            # <c A> = c<A> where c is a commutative term
            A = A.expand()
            cA, ncA = A.args_cnc()
            return Mul(Mul(*cA), Expectation(Mul._from_args(ncA), self.is_normal_order).expand())
        
        if isinstance(A, Integral):
            # <∫adx> ->  ∫<a>dx
            func, lims = A.function, A.limits
            new_args = [Expectation(func, self.is_normal_order).expand()]
            for lim in lims:
                new_args.append(lim)
            return Integral(*new_args)
        
        return self
    
    def doit(self, **hints):
        """
        return the expectation value normally ordered operator if is_normal_order=True
        """
        if self.is_normal_order == True:
            return Expectation(normal_order(self.args[0]), False)
        return self
    
    def eval_state(self, state):
        return qapply(Dagger(state) * self.args[0] * state, dagger=True).doit()
        
    def _latex(self, printer, *args):
        if self.is_normal_order:
            return r"\left\langle: %s :\right\rangle" % printer._print(self.args[0], *args)
        else:
            return r"\left\langle %s \right\rangle" % printer._print(self.args[0], *args)

#-----------------------------------------------------------------------------
# Covariance
#-----------------------------------------------------------------------------

class Covariance(Expr):
    """Covariance of two operators, expressed in terms of bracket <A, B>
    
    If the third argument, 'is_normal_order' is 'True',
    the normal ordering notation (<: , :>) is attached.

    doit() returns the expression in terms of expectation values.
        < A, B > --> < AB > - < A >< B >

    Parameters
    ==========
    
    A : Expr
        The first argument of the expectation value

    B : Expr
        The second argument of the expectation value
    
    is_normal_order : bool
        A bool that indicates if the operator inside the Expectation
        value bracket should be normally ordered (True) or left
        untouched (False, default value)

    Examples
    ========

    >>> A, B = Operator("A"), Operator("B")
    >>> Covariance(A, B)
    < A, B >
    >>> Covariance(A, B, True)
    <:A, B:>
    >>> Covariance(A, B).doit()
    < AB > - < A >< B >
    >>> Covariance(A, B, True).doit()
    <:AB:> - <:A:><:B:>
    
    """

    is_commutative = True
    @property    
    def is_normal_order(self):
        return bool(self.args[2])
    
    @classmethod
    def default_args(self):
        return (Symbol("A"), Symbol("B"), False)

    def __new__(cls, *args, **hints):
        if not len(args) in [2, 3]:
            raise ValueError('2 or 3 parameters expected, got %s' % args)

        if len(args) == 2:
            args = (args[0], args[1], Integer(0))

        if len(args) == 3:
            args = (args[0], args[1], Integer(args[2]))

        return Expr.__new__(cls, *args)
    
    def _eval_expand_covariance(self, **hints):        
        A, B = self.args[0], self.args[1]
        # <A + B, C> = <A, C> + <B, C>
        if isinstance(A, Add):
            return Add(*(Covariance(a, B, self.is_normal_order).expand()
                         for a in A.args))

        # <A, B + C> = <A, B> + <A, C>
        if isinstance(B, Add):
            return Add(*(Covariance(A, b, self.is_normal_order).expand()
                         for b in B.args))
        
        if isinstance(A, Mul):
            A = A.expand()            
            cA, ncA = A.args_cnc()
            return Mul(Mul(*cA), Covariance(Mul._from_args(ncA), B, 
                                            self.is_normal_order).expand())
        if isinstance(B, Mul):
            B = B.expand()            
            cB, ncB = B.args_cnc()
            return Mul(Mul(*cB), Covariance(A, Mul._from_args(ncB),
                                             self.is_normal_order).expand())        
        if isinstance(A, Integral):
            # <∫adx, B> ->  ∫<a, B>dx
            func, lims = A.function, A.limits
            new_args = [Covariance(func, B, self.is_normal_order).expand()]
            for lim in lims:
                new_args.append(lim)
            return Integral(*new_args)
        if isinstance(B, Integral):
            # <A, ∫bdx> ->  ∫<A, b>dx
            func, lims = B.function, B.limits
            new_args = [Covariance(A, func, self.is_normal_order).expand()]
            for lim in lims:
                new_args.append(lim)
            return Integral(*new_args)
        return self
    
    def doit(self, **hints):
        """ Evaluate covariance of two operators A and B """
        A = self.args[0]
        B = self.args[1]
        no = self.is_normal_order
        return Expectation(A*B, no) - Expectation(A, no) * Expectation(B, no)
    
    def _latex(self, printer, *args):
        if self.is_normal_order:
            return r"\left\langle: %s, %s :\right\rangle" % tuple([
                printer._print(self.args[0], *args),
                printer._print(self.args[1], *args)])
        else:
            return r"\left\langle %s, %s \right\rangle" % tuple([
                printer._print(self.args[0], *args),
                printer._print(self.args[1], *args)])
