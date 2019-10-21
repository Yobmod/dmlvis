"""
Numpy's mypy stub. Only type declarations for ndarray, the scalar hierarchy and array creation
methods are provided.
"""

from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, Type,
                    TypeVar, Union, overload)
import abc
from pathlib import Path


class dtype:
    ...


_dtype = dtype


class flagsobj:
    """numpy.flagsobj"""
    aligned = None       # type: bool
    behaved = None       # type: bool
    c_contiguous = None  # type: bool
    carray = None        # type: bool
    contiguous = None    # type: bool
    f_contiguous = None  # type: bool
    farray = None        # type: bool
    fnc = None           # type: bool
    forc = None          # type: bool
    fortran = None       # type: bool
    owndata = None       # type: bool
    updateifcopy = None  # type: bool
    writeable = None     # type: bool
    def __getitem__(self, item: str) -> bool: ...
    def __setitem__(self, item: str, value: bool) -> None: ...

#
# Type variables. _T wasn't used to avoid confusions with ndarray's "T" attribute.
#


_S = TypeVar('_S')
_U = TypeVar('_U')
_V = TypeVar('_V')

#
# Auxiliary types
#

ShapeType = Union[int, Tuple[int, ...]]
AxesType = Union[int, Tuple[int, ...]]
OrderType = Union[str, Sequence[str]]
DtypeType = Union[dtype, type]


class flatiter(Generic[_S], Iterator[_S], metaclass=abc.ABCMeta):
    coords = ...  # type: ShapeType
    def copy(self) -> 'flatiter[_S]': ...


class _ArrayLike(Generic[_S]):
    """
    "array-like" interface that both numpy.ndarray and all scalars (descendants of numpy.generic)
    implement this interface.
    """
    #
    # Array-like structures attributes
    #
    T = None         # type: _ArrayLike[_S]
    data = None      # type: Any
    dtype = None     # type: _dtype
    flags = None     # type: flagsobj
    flat = None      # type: flatiter[_ArrayLike[_S]]
    imag = None      # type: _ArrayLike[_S]
    real = None      # type: _ArrayLike[_S]
    size = None      # type: int
    itemsize = None  # type: int
    nbytes = None    # type: int
    ndim = None      # type: int
    shape = None     # type: Tuple[int, ...]
    strides = None   # type: Tuple[int, ...]
    base = None      # type: Optional[_ArrayLike[_S]]

    #
    # Array-like methods
    #

    # Once this issue https://github.com/python/mypy/issues/1907 is resolved, most methods that
    # have an 'out' argument, will be implemented using overload instead of with a Union
    # result. mypy is smart enough to assign the proper type (_ArrayLike[_U]) when out is present
    # but it falls back to the union when it's not.
    def all(self, axis: AxesType = None, out: '_ArrayLike[_U]' = None,
            keepdims: bool = False) -> Union['_ArrayLike[_U]', '_ArrayLike[bool]']: ...

    def any(self, axis: AxesType = None, out: '_ArrayLike[_U]' = None,
            keepdims: bool = False) -> Union['_ArrayLike[_U]', '_ArrayLike[bool]']: ...

    def argmax(self, axis: int = None,
               out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_U]', '_ArrayLike[int]']: ...

    def argmin(self, axis: int = None,
               out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_U]', '_ArrayLike[int]']: ...

    def argpartition(self, kth: Union[int, Sequence[int]], axis: Optional[int] = -1,
                     kind: str = 'introselect', order: OrderType = None) -> '_ArrayLike[int]': ...

    def argsort(self, axis: int = None, kind: str = 'quicksort',
                order: OrderType = None) -> '_ArrayLike[int]': ...

    def astype(self, dtype: Any, order: str = 'K', casting: str = 'unsafe', subok: bool = True,
               copy: bool = False) -> '_ArrayLike[Any]': ...

    def byteswap(self, inplace: bool = False) -> '_ArrayLike[_S]': ...

    def choose(self, choices: Sequence['_ArrayLike[_V]'], out: '_ArrayLike[_U]' = None,
               mode: str = 'raise') -> Union['_ArrayLike[_U]', '_ArrayLike[_V]']: ...

    def clip(self, a_min: Any, a_max: Any,
             out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def compress(self, condition: Sequence[bool], axis: int = None,
                 out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def conj(self) -> '_ArrayLike[_S]': ...

    def conjugate(self) -> '_ArrayLike[_S]': ...

    def copy(self, order: str = 'C') -> '_ArrayLike[_S]': ...

    def cumprod(self, axis: int = None, dtype: Any = None,
                out: '_ArrayLike[Any]' = None) -> '_ArrayLike[Any]': ...

    def cumsum(self, axis: int = None, dtype: DtypeType = None,
               out: '_ArrayLike[Any]' = None) -> '_ArrayLike[Any]': ...

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> '_ArrayLike[_S]':
        ...

    def dot(self, b: '_ArrayLike[Any]', out: '_ArrayLike[Any]' = None) -> '_ArrayLike[Any]':
        ...

    def dump(self, file: str) -> None: ...

    def dumps(self) -> str: ...

    def fill(self, value: _S) -> None: ...

    def flatten(self, order: str = 'C') -> '_ArrayLike[_S]': ...

    def getfield(self, dtype: DtypeType, offset: int = 0) -> '_ArrayLike[Any]':
        ...

    def item(self, args: AxesType) -> 'generic': ...

    def itemset(self, arg0: Union[int, Tuple[int, ...]],
                arg1: Any = None) -> None: ...

    def max(self, axis: AxesType = None,
            out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def mean(self, axis: AxesType = None, dtype: Any = None,
             out: '_ArrayLike[_U]' = None, keepdims: bool = False) -> '_ArrayLike[floating]': ...

    def min(self, axis: AxesType = None,
            out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def newbyteorder(self, new_order: str = 'S') -> '_ArrayLike[_S]': ...

    def nonzero(self) -> '_ArrayLike[int]': ...

    def partition(self, kth: AxesType, axis: int = -1, kind: str = 'introselect',
                  order: OrderType = None) -> None: ...

    def prod(self, axis: AxesType = None, dtype: DtypeType = None,
             out: '_ArrayLike[_U]' = None, keepdims: bool = False) -> '_ArrayLike[Any]': ...

    def ptp(self, axis: int = None,
            out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def put(self, ind: '_ArrayLike[int]',
            v: '_ArrayLike[_S]', mode: str = 'raise') -> None: ...

    def ravel(self, order: str = 'C') -> '_ArrayLike[_S]': ...

    def repeat(self, repeats: Union[int, Sequence[int]],
               axis: int = None) -> '_ArrayLike[_S]': ...

    def reshape(self, newshape: ShapeType,
                order: str = 'C') -> '_ArrayLike[_S]': ...

    def resize(self, new_shape: ShapeType, refcheck: bool = True) -> None: ...

    def round(self, decimals: int = 0,
              out: '_ArrayLike[_U]' = None) -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def searchsorted(self, v: Union[_S, '_ArrayLike[_S]'], side: str = 'left',
                     sorter: '_ArrayLike[int]' = None) -> '_ArrayLike[int]': ...

    def setfield(self, val: Any, dtype: DtypeType,
                 offset: int = 0) -> None: ...

    def setflags(self, write: bool = None, align: bool = None,
                 uic: bool = None) -> None: ...

    def sort(self, axis: int = -1, kind: str = 'quicksort',
             order: OrderType = None) -> None: ...

    def squeeze(self, axis: AxesType = None) -> '_ArrayLike[_S]': ...

    def std(self, axis: AxesType = None, dtype: DtypeType = None,
            out: '_ArrayLike[_U]' = None, ddof: int = 0, keepdims: bool = False) -> '_ArrayLike[floating]': ...

    def sum(self, axis: AxesType = None, dtype: DtypeType = None,
            out: '_ArrayLike[_U]' = None,
            keepdims: bool = False) -> '_ArrayLike[Any]': ...

    def swapaxes(self, axis1: int, axis2: int) -> '_ArrayLike[_S]': ...

    def take(self, indices: Sequence[int], axis: int = None,
             out: '_ArrayLike[_U]' = None,
             mode: str = 'raise') -> Union['_ArrayLike[_S]', '_ArrayLike[_U]']: ...

    def tobytes(self, order: str = 'C') -> bytes: ...

    def tofile(self, fid: object, sep: str = '',  # TODO fix fid definition (bug in mypy io's namespace https://github.com/python/mypy/issues/1462)
               format: str = '%s') -> None: ...

    def tolist(self) -> List[Any]: ...

    def tostring(self, order: str = 'C') -> bytes: ...

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1,
              dtype: DtypeType = None, out: '_ArrayLike[_U]' = None) -> '_ArrayLike[Any]': ...

    def transpose(self, axes: Optional[AxesType] = None) -> '_ArrayLike[_S]': ...

    def var(self, axis: AxesType = None, dtype: DtypeType = None,
            out: '_ArrayLike[_U]' = None, ddof: int = 0, keepdims: bool = False) -> '_ArrayLike[Any]': ...

    def view(self, dtype: Union[DtypeType, Type['ndarray']] = None,
             type: type = None) -> '_ArrayLike[Any]': ...

    #
    # Magic methods
    #

    def __abs__(self) -> '_ArrayLike[_S]': ...

    def __add__(self, value: object) -> '_ArrayLike[Any]': ...

    def __and__(self, value: object) -> '_ArrayLike[int]': ...

    def __array__(self, dtype: DtypeType = None) -> '_ArrayLike[Any]': ...

    def __array_prepare__(self, context: object = None) -> '_ArrayLike[Any]':
        ...

    def __array_wrap__(self, context: object = None) -> '_ArrayLike[Any]': ...

    def __bool__(self) -> bool: ...

    def __complex__(self) -> complex: ...

    def __contains__(self, key: object) -> bool: ...

    def __copy__(self) -> '_ArrayLike[_S]': ...

    def __deepcopy__(self) -> '_ArrayLike[_S]': ...

    def __delattr__(self, name: str) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def __dir__(self) -> List[str]: ...

    def __divmod__(
        self, value: object) -> Tuple['_ArrayLike[int]', '_ArrayLike[float]']: ...

    def __eq__(self, value: object) -> '_ArrayLike[bool]': ...  # type: ignore

    def __float__(self) -> float: ...

    def __floordiv__(self, value: object) -> '_ArrayLike[int]': ...

    def __ge__(self, value: object) -> '_ArrayLike[bool]': ...

    def __getattribute__(self, name: str) -> Any: ...

    def __getitem__(self, key: Any) -> '_ArrayLike[_S]': ...

    def __gt__(self, value: object) -> '_ArrayLike[bool]': ...

    def __iadd__(self, value: object) -> None: ...

    def __iand__(self, value: object) -> None: ...

    def __ifloordiv__(self, value: object) -> None: ...

    def __ilshift__(self, value: object) -> None: ...

    def __imatmul__(self, value: '_ArrayLike[Any]') -> None: ...

    def __imod__(self, value: object) -> None: ...

    def __imul__(self, value: object) -> None: ...

    def __index__(self) -> int: ...

    def __int__(self) -> int: ...

    def __invert__(self) -> '_ArrayLike[_S]': ...

    def __ior__(self, value: object) -> None: ...

    def __ipow__(self, value: object) -> None: ...

    def __irshift__(self, value: object) -> None: ...

    def __isub__(self, value: object) -> None: ...

    def __iter__(self) -> Iterator['_ArrayLike[_S]']: ...

    def __itruediv__(sel, value: object) -> None: ...

    def __ixor__(self, value: object) -> None: ...

    def __le__(self, value: object) -> '_ArrayLike[bool]': ...

    def __len__(self) -> int: ...

    def __lshift__(self, value: object) -> '_ArrayLike[_S]': ...

    def __lt__(self, value: object) -> '_ArrayLike[bool]': ...

    def __matmul__(self, value: '_ArrayLike[Any]') -> '_ArrayLike[Any]': ...

    def __mod__(self, value: object) -> '_ArrayLike[_S]': ...

    def __mul__(self, value: object) -> '_ArrayLike[Any]': ...

    def __ne__(self, value: object) -> '_ArrayLike[bool]': ...  # type: ignore

    def __neg__(self) -> '_ArrayLike[_S]': ...

    def __or__(self, value: object) -> '_ArrayLike[_S]': ...

    def __pos__(self) -> '_ArrayLike[_S]': ...

    def __pow__(self, value: object) -> '_ArrayLike[Any]': ...

    def __radd__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rand__(self, value: object) -> '_ArrayLike[_S]': ...

    def __rdivmod__(
        self, value: object) -> Tuple['_ArrayLike[int]', '_ArrayLike[float]']: ...

    def __rfloordiv__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rlshift__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rmatmul__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rmod__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rmul__(self, value: object) -> '_ArrayLike[Any]': ...

    def __ror__(self, value: object) -> '_ArrayLike[_S]': ...

    def __rpow__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rrshift__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rshift__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rsub__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rtruediv__(self, value: object) -> '_ArrayLike[Any]': ...

    def __rxor__(self, value: object) -> '_ArrayLike[_S]': ...

    def __setattr__(self, name: str, value: Any) -> None: ...

    def __setitem__(self, key: Any, value: Any) -> None: ...

    def __str__(self) -> str: ...

    def __sub__(self, value: object) -> '_ArrayLike[Any]': ...

    def __truediv__(sel, value: object) -> '_ArrayLike[Any]': ...

    def __xor__(self, value: object) -> '_ArrayLike[_S]': ...

#
# numpy's scalar hierarchy (http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#scalars)
#


class generic(_ArrayLike[_S], Generic[_S]):
    ...


class bool_(generic[bool]):
    ...


bool8 = bool_


class object_(generic[Any]):
    ...


class number(generic[_S], Generic[_S]):
    ...


class integer(number[int]):
    ...


class signedinteger(integer):
    ...


class byte(signedinteger):
    ...


class short(signedinteger):
    ...


class intc(signedinteger):
    ...


class int_(signedinteger):
    ...


class longlong(signedinteger):
    ...


class int8(signedinteger):
    ...


class int16(signedinteger):
    ...


class int32(signedinteger):
    ...


class int64(signedinteger):
    ...


class unsignedinteger(integer):
    ...


class ubyte(unsignedinteger):
    ...


class ushort(unsignedinteger):
    ...


class uintc(unsignedinteger):
    ...


class uint(unsignedinteger):
    ...


class ulonglong(unsignedinteger):
    ...


class uint8(signedinteger):
    ...


class uint16(signedinteger):
    ...


class uint32(signedinteger):
    ...


class uint64(signedinteger):
    ...


class inexact(number[float]):
    ...


class floating(inexact):
    ...


class half(floating):
    ...


class single(floating):
    ...


class float_(floating):
    ...


class longfloat_(floating):
    ...


class float16(floating):
    ...


class float32(floating):
    ...


class float64(floating):
    ...


class float128(floating):
    ...


class complexfloating(inexact):
    ...


class csingle(complexfloating):
    ...


class complex_(complexfloating):
    ...


class clongfloat(complexfloating):
    ...


class complex64(complexfloating):
    ...


class complex128(complexfloating):
    ...


class complex256(complexfloating):
    ...


class flexible(generic[_S], Generic[_S]):
    ...


class character(flexible[str]):
    ...


class str_(character):
    ...


class unicode_(character):
    ...


class void(flexible[None]):
    ...


class ndarray(_ArrayLike[_S], Generic[_S]):
    """numpy.ndarray"""
    ctypes = None    # type: Any  # TODO Implement ctypes type hint

    # TODO Need to find a way to restrict buffer type
    def __init__(self, shape: Tuple[int, ...], dtype: DtypeType = None,
                 buffer: Any = None, offset: int = None,
                 strides: Tuple[int, ...] = None, order: str = None) -> None: ...

#
# Array creation routines
#


class random():
    def __init__(self) -> None: ...

    @staticmethod
    def seed(seed: Optional[Union[int, _ArrayLike]] = None) -> None: ...

    @staticmethod
    def RandomState(seed: Optional[Union[int, _ArrayLike]] = None) -> None: ...

    @staticmethod
    @overload
    def rand(size: int) -> float: ...

    @staticmethod
    @overload
    def rand(size: Tuple[int, ...]) -> ndarray[float]: ...

    @staticmethod
    @overload
    def rand(size: None) -> float: ...


@overload  # 1 dimensiosnal arrays give in int
def argmax(input: Sequence[Any], axis: Optional[int] = None, out: Optional[ndarray[Any]] = None) -> int: ...


@overload  # 1 dimensiosnal arrays give in int
def argmax(input: _ArrayLike[Any], axis: Optional[int] = None, out: Optional[ndarray[Any]] = None) -> Union[int, ndarray[int]]: ...


def array(object: Any, dtype: Any = None, copy: bool = True,
          order: str = None, subok: bool = False,
          ndmin: int = 0) -> ndarray[Any]: ...


def asarray(a: Any, dtype: DtypeType = None,
            order: str = None) -> ndarray[Any]: ...


def asanyarray(a: Any, dtype: DtypeType = None,
               order: str = None) -> ndarray[Any]: ...  # TODO figure out a way to restrict the return type


def asmatrix(
    data: Any, dtype: DtypeType = None) -> Any: ...  # TODO define matrix


def ascontiguousarray(a: Any, dtype: DtypeType = None) -> ndarray[Any]: ...


def concatenate(inputs: Sequence[Union[_ArrayLike[Any], Sequence[Any]]], axis: int = 0, out: Optional[ndarray[Any]] = None) -> ndarray[Any]: ...


def copy(a: Any, order: str = None) -> ndarray[Any]: ...


def empty(shape: ShapeType, dtype: DtypeType = float,
          order: str = 'C') -> ndarray[Any]: ...


def empty_like(a: Any, dtype: Any = None, order: str = 'K',
               subok: bool = True) -> ndarray[Any]: ...


def expand_dims(array: _ArrayLike[Any], axis: int) -> ndarray[_ArrayLike[Any]]: ...


def eye(N: int, M: int = None, k: int = 0,
        dtype: DtypeType = float) -> ndarray[Any]: ...


def frombuffer(buffer: Any, dtype: DtypeType = float, count: int = -1,  # TODO figure out a way to restrict buffer
               offset: int = 0) -> ndarray[Any]: ...


# TODO fix file definition (There's a bug in mypy io's namespace https://github.com/python/mypy/issues/1462)
def fromfile(file: object, dtype: DtypeType = float,
             count: int = -1, sep: str = '') -> ndarray[Any]: ...


def full(shape: ShapeType, fill_value: Any, dtype: DtypeType = None,
         order: str = 'C') -> ndarray[Any]: ...


def full_like(a: Any, fill_value: Any, dtype: DtypeType = None, order: str = 'C',
              subok: bool = True) -> ndarray[Any]: ...


def fromfunction(function: Callable[..., _S], shape: ShapeType,
                 dtype: DtypeType = float) -> ndarray[_S]: ...


def fromiter(iterable: Iterator[Any], dytpe: DtypeType,
             count: int = -1) -> ndarray[Any]: ...


def fromstring(string: str, dtype: DtypeType = float,
               count: int = -1, sep: str = '') -> ndarray[Any]: ...


def identity(n: int, dtype: DtypeType = None) -> ndarray[Any]: ...


def loadtxt(fname: Any, dtype: DtypeType = float, comments: Union[str, Sequence[str]] = '#',
            delimiter: str = None, converters: Dict[int, Callable[[Any], float]] = None,
            skiprows: int = 0, usecols: Sequence[int] = None,
            unpack: bool = False, ndmin: int = 0) -> ndarray[float]: ...


def ones(shape: ShapeType,
         dtype: Optional[DtypeType] = ..., order: str = 'C') -> ndarray[Any]: ...


def ones_like(a: Any, dtype: Any = None, order: str = 'K',
              subok: bool = True) -> ndarray[Any]: ...


def zeros(shape: ShapeType, dtype: DtypeType = float,
          order: str = 'C') -> ndarray[Any]: ...


def zeros_like(a: Any, dtype: Any = None, order: str = 'K',
               subok: bool = True) -> ndarray[Any]: ...


def savetxt(filename: Union[str, Path], array: _ArrayLike, delimiter: Optional[str] = ..., header: Optional[str] = ...) -> None: ...


# Specific values
inf: float
