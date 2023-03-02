from jax._src.numpy.vectorize import (
  functools,
  _apply_excluded,
  map,
  jnp,
  _parse_gufunc_signature,
  _parse_input_dimensions,
  _check_output_dims,
  zip,
  api,
)

def vectorize(pyfunc, *, excluded=frozenset(), signature=None):
  """Define a vectorized function with broadcasting.

  :func:`vectorize` is a convenience wrapper for defining vectorized
  functions with broadcasting, in the style of NumPy's
  `generalized universal functions <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.
  It allows for defining functions that are automatically repeated across
  any leading dimensions, without the implementation of the function needing to
  be concerned about how to handle higher dimensional inputs.

  :func:`jax.numpy.vectorize` has the same interface as
  :class:`numpy.vectorize`, but it is syntactic sugar for an auto-batching
  transformation (:func:`vmap`) rather than a Python loop. This should be
  considerably more efficient, but the implementation must be written in terms
  of functions that act on JAX arrays.

  Args:
    pyfunc: function to vectorize.
    excluded: optional set of integers representing positional arguments for
      which the function will not be vectorized. These will be passed directly
      to ``pyfunc`` unmodified.
    signature: optional generalized universal function signature, e.g.,
      ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If
      provided, ``pyfunc`` will be called with (and expected to return) arrays
      with shapes given by the size of corresponding core dimensions. By
      default, pyfunc is assumed to take scalars arrays as input and output.

  Returns:
    Vectorized version of the given function.

  Here are a few examples of how one could write vectorized linear algebra
  routines using :func:`vectorize`:

  >>> from functools import partial

  >>> @partial(jnp.vectorize, signature='(k),(k)->(k)')
  ... def cross_product(a, b):
  ...   assert a.shape == b.shape and a.ndim == b.ndim == 1
  ...   return jnp.array([a[1] * b[2] - a[2] * b[1],
  ...                     a[2] * b[0] - a[0] * b[2],
  ...                     a[0] * b[1] - a[1] * b[0]])

  >>> @partial(jnp.vectorize, signature='(n,m),(m)->(n)')
  ... def matrix_vector_product(matrix, vector):
  ...   assert matrix.ndim == 2 and matrix.shape[1:] == vector.shape
  ...   return matrix @ vector

  These functions are only written to handle 1D or 2D arrays (the ``assert``
  statements will never be violated), but with vectorize they support
  arbitrary dimensional inputs with NumPy style broadcasting, e.g.,

  >>> cross_product(jnp.ones(3), jnp.ones(3)).shape
  (3,)
  >>> cross_product(jnp.ones((2, 3)), jnp.ones(3)).shape
  (2, 3)
  >>> cross_product(jnp.ones((1, 2, 3)), jnp.ones((2, 1, 3))).shape
  (2, 2, 3)
  >>> matrix_vector_product(jnp.ones(3), jnp.ones(3))  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ValueError: input with shape (3,) does not have enough dimensions for all
  core dimensions ('n', 'k') on vectorized function with excluded=frozenset()
  and signature='(n,k),(k)->(k)'
  >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones(3)).shape
  (2,)
  >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones((4, 3))).shape
  (4, 2)

  Note that this has different semantics than `jnp.matmul`:

  >>> jnp.matmul(jnp.ones((2, 3)), jnp.ones((4, 3)))  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  TypeError: dot_general requires contracting dimensions to have the same shape, got [3] and [4].
  """
  if any(not isinstance(exclude, int) for exclude in excluded):
    raise TypeError("jax.numpy.vectorize can only exclude integer arguments, "
                    "but excluded={!r}".format(excluded))
  if excluded and min(excluded) < 0:
    raise ValueError(f"excluded={excluded!r} contains negative numbers")

  @functools.wraps(pyfunc)
  def wrapped(*args):
    error_context = ("on vectorized function with excluded={!r} and "
                     "signature={!r}".format(excluded, signature))
    excluded_func, args = _apply_excluded(pyfunc, excluded, args)
    args = tuple(map(jnp.asarray, args))

    if signature is not None:
      input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
    else:
      input_core_dims = [()] * len(args)
      output_core_dims = None

    broadcast_shape, dim_sizes = _parse_input_dimensions(
        args, input_core_dims, error_context)

    checked_func = _check_output_dims(
        excluded_func, dim_sizes, output_core_dims, error_context)

    # Rather than broadcasting all arguments to full broadcast shapes, prefer
    # expanding dimensions using vmap. By pushing broadcasting
    # into vmap, we can make use of more efficient batching rules for
    # primitives where only some arguments are batched (e.g., for
    # lax_linalg.triangular_solve), and avoid instantiating large broadcasted
    # arrays.

    squeezed_args = []
    rev_filled_shapes = []

    for arg, core_dims in zip(args, input_core_dims):
      noncore_shape = arg.shape[:arg.ndim - len(core_dims)]
      
      pad_ndim = len(broadcast_shape) - len(noncore_shape)
      filled_shape = pad_ndim * (1,) + noncore_shape
      assert len(filled_shape) == len(broadcast_shape)
      assert jnp.broadcast_shapes(broadcast_shape, filled_shape) == broadcast_shape
      rev_filled_shapes.append(filled_shape[::-1])

      squeeze_indices = tuple(i for i, size in enumerate(noncore_shape) if size == 1)
      squeezed_arg = jnp.squeeze(arg, axis=squeeze_indices)
      squeezed_args.append(squeezed_arg)

    vectorized_func = checked_func
    dims_to_expand = []
    for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes)):
      in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
      if all(axis is None for axis in in_axes):
        dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
      else:
        vectorized_func = api.vmap(vectorized_func, in_axes)
    result = vectorized_func(*squeezed_args)
    return jnp.expand_dims(result, axis=dims_to_expand)

  return wrapped
