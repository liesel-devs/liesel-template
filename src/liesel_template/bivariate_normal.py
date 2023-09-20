import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.internal import parameter_properties


class BivariateNormal(tfd.MultivariateNormalTriL):
    """A bivariate normal TensorFlow Probability distribution for RLiesel."""

    def __init__(
        self,
        loc1=0.0,
        loc2=0.0,
        scale1=1.0,
        scale2=1.0,
        cor=0.0,
        validate_args=False,
        allow_nan_stats=True,
        experimental_use_kahan_sum=False,
        name="BivariateNormal",
    ):
        parameters = dict(locals())

        loc1, loc2, scale1, scale2, cor = jnp.broadcast_arrays(
            loc1,
            loc2,
            scale1,
            scale2,
            cor,
        )

        loc = jnp.stack([loc1, loc2], axis=-1)

        tril11 = scale1
        tril12 = 0.0 * scale1
        tril21 = cor * scale2
        tril22 = jnp.sqrt(scale2**2 - tril21**2)
        tril1 = jnp.stack([tril11, tril12], axis=-1)
        tril2 = jnp.stack([tril21, tril22], axis=-1)
        tril = jnp.stack([tril1, tril2], axis=-2)

        super().__init__(
            loc=loc,
            scale_tril=tril,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            experimental_use_kahan_sum=experimental_use_kahan_sum,
            name=name,
        )

        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {
            "loc1": parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Identity(),
            ),
            "loc2": parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Identity(),
            ),
            "scale1": parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Exp(),
            ),
            "scale2": parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Exp(),
            ),
            "cor": parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Tanh(),
            ),
        }
