from abc import abstractmethod, ABC

from torch import Tensor


class LinearOperator(ABC):
    """A function that linearly maps an input to an output.

    This object represents the matrix A in the operation y = A * x,
    where x is the input vector and y is the output vector.  Subclasses
    need to implement two methods: (1) "apply", which computes A * x
    and (2) "transpose", which computes A^T * y.  These methods should
    be able to parallelize computation across multiple inputs/outputs
    for batch computation.  Additionally, subclasses need to implement
    two properties -- "inp_dim" and "out_dim" -- which denote the
    dimensionality of x and y, respectively.

    """

    @abstractmethod
    def apply(self, inp: Tensor) -> Tensor:
        """Applies the operator to an input.

        Given x, this function returns y = A * x.

        Every linear operator should implement this method, but its
        functionality is accessed through the __call__ method.

        Parameters
        ----------
        inp : Tensor
            Input with shape (batch_size x input_dim).

        Returns
        -------
        Tensor
            Operator times input with shape (batch_size x output_dim).

        """
        pass

    @abstractmethod
    def transpose(self, out: Tensor) -> Tensor:
        """Applies the transpose of the operator to an output.

        Given y, this function returns z = A^T * y.  If A
        is an orthogonal matrix, then this is equivalent to undoing
        the "apply" method with x = A^{-1} * y.

        Every linear operator should implement this method, but its
        functionality is accessed through the T method.

        Parameters
        ----------
        out : Tensor
            Output with shape (batch_size x output_dim).

        Returns
        -------
        Tensor
            Transpose times output with shape (batch_size x input_dim).

        """
        pass

    @property
    @abstractmethod
    def inp_dim(self) -> int:
        """int: The dimensionality of the input space."""
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """int: The dimensionality of the output space."""
        pass

    def __call__(self, inp: Tensor) -> Tensor:
        """Checks input shape and applies the operator.

        Parameters
        ----------
        inp : Tensor
            An input with shape (batch_size x input_dim).

        Returns
        -------
        Tensor
            Operator times input with shape (batch_size x output_dim).

        """
        assert inp.shape[-1] == self.inp_dim
        return self.apply(inp)

    def T(self, out: Tensor) -> Tensor:
        """Checks output shape and applies the operator's transpose.

        Parameters
        ----------
        out : Tensor
            An output with shape (batch_size x output_dim).

        Returns
        -------
        Tensor
            Transpose times output with shape (batch_size x input_dim).

        """
        assert out.shape[-1] == self.out_dim
        return self.transpose(out)

    def __repr__(self) -> str:
        """Gets the string representation of the linear operator.

        Returns
        -------
        str
            The string representation.

        """
        return self.__class__.__name__
