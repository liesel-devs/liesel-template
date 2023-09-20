import jax.numpy as jnp

from liesel_template import BivariateNormal

bn1 = BivariateNormal(
    loc1=1.0,
    loc2=3.0,
    scale1=5.0,
    scale2=7.0,
    cor=0.3,
)

bn2 = BivariateNormal(
    loc1=jnp.zeros(2),
    loc2=jnp.zeros(2),
    scale1=jnp.ones(2),
    scale2=jnp.ones(2),
    cor=jnp.zeros(2),
)

bn3 = BivariateNormal(
    loc1=jnp.zeros([2, 2]),
    loc2=jnp.zeros([2, 2]),
    scale1=jnp.ones([2, 2]),
    scale2=jnp.ones([2, 2]),
    cor=jnp.zeros([2, 2]),
)

bn4 = BivariateNormal(
    loc1=jnp.zeros(2),
    loc2=0.0,
    scale1=jnp.ones(2),
    scale2=jnp.ones(2),
    cor=0.0,
)


def test_shapes():
    assert bn1.event_shape == 2
    assert bn2.event_shape == 2
    assert bn3.event_shape == 2
    assert bn4.event_shape == 2

    assert bn1.batch_shape == []
    assert bn2.batch_shape == [2]
    assert bn3.batch_shape == [2, 2]
    assert bn4.batch_shape == [2]


def test_log_prob():
    zeros = jnp.array([0.0, 0.0])

    assert jnp.allclose(bn1.log_prob(zeros), -5.44071)
    assert jnp.allclose(bn2.log_prob(zeros), -1.837877 * jnp.ones(2))
    assert jnp.allclose(bn3.log_prob(zeros), -1.837877 * jnp.ones([2, 2]))
    assert jnp.allclose(bn4.log_prob(zeros), -1.837877 * jnp.ones(2))
