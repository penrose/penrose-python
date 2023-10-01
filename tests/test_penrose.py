import jax
import jax.numpy as jnp
from pytest import approx


def test_answer():
    assert jax.grad(jnp.sin)(1.0) == approx(jnp.cos(1.0))
