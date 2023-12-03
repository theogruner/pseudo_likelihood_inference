from typing import List, Dict, Callable
import numpy as np

import jax.numpy as jnp
from jaxtyping import Array
from diffrax import ODETerm, diffeqsolve, SaveAt, Tsit5


from pli.figures import TrajectoryFigure
from pli.models.basic.uniform import UniformDistribution
from .utils import support_dict_to_array
from .base import Task


class Furuta(Task):
    name = "Furuta"

    def __init__(
        self,
        dt: float = 0.01,  # 1000 Hz
        seq_len: int = 500,
        sampling_frequency: int = 1,
        n_train_data=50,
        n_eval_data=50,
        n_posterior_samples=1000,
        n_chains=1,
        n_warmup=1000,
        **_ignore
    ):
        super().__init__(
            n_train_data,
            n_eval_data,
            n_posterior_samples,
            n_chains,
            n_warmup,
        )
        self.dt = dt
        self.seq_len = seq_len
        self.sampling_frequency = sampling_frequency

    def get_simulator(self) -> Callable:
        def simulator(rng_key, param) -> jnp.ndarray:
            init_state = self.sample_init_state(rng_key, ())
            domain_param_dict = self.make_domain_param_dict(jnp.asarray(param))
            constants = Furuta.calc_constants(domain_param_dict)

            km = domain_param_dict["motor_back_emf"]
            Rm = domain_param_dict["motor_resistance"]
            Dr = domain_param_dict["damping_rot_pole"]
            Dp = domain_param_dict["damping_pend_pole"]

            def dyn(t, s, args):
                # Decompose state
                # th = s[0]
                al = s[1]
                thd = s[2]
                ald = s[3]

                # calculate mass matrix
                m11 = constants[0] + constants[1] * (jnp.sin(al) ** 2)
                m12 = constants[2] * jnp.cos(al)
                m22 = constants[3]
                det_m = m11 * m22 - m12 ** 2

                u = 0.0
                trq = km * (u - km * thd) / Rm
                # trq = jnp.zeros_like(m11)
                c0 = constants[1] * jnp.sin(2 * al) * thd * ald - constants[
                                                                        2
                                                                     ] * jnp.sin(al) * (ald ** 2)
                c1 = -0.5 * constants[1] * jnp.sin(2 * al) * (
                        thd ** 2
                ) + constants[4] * jnp.sin(al)

                x = trq - Dr * thd - c0
                y = -Dp * ald - c1

                # compute accelerations
                thdd = (m22 * x - m12 * y) / det_m
                aldd = (m11 * y - m12 * x) / det_m
                return jnp.stack([thd, ald, thdd, aldd], axis=-1)

            t = jnp.linspace(
                0,
                self.seq_len * self.dt * self.sampling_frequency,
                self.seq_len + 1,
            )[:-1]
            term = ODETerm(dyn)
            solver = Tsit5()
            sol = diffeqsolve(
                term,
                solver,
                t0=0,
                t1=self.seq_len * self.dt * self.sampling_frequency,
                dt0=self.dt,
                y0=init_state,
                saveat=SaveAt(ts=t),
            )
            # states = jnp.transpose(sol.ys, (1, 0, 2))
            return Furuta.observation(sol.ys)

        return simulator

    @staticmethod
    def sample_init_state(rng_key, shape):
        """Sample initial state from uniform distribution.
        The pendulum is slightly perturbed around its unstable equilibrium
        and has a small random initial angular velocity."""
        min_init_state = (
            jnp.array([-5.0, 175.0, -0.1, -0.1]) / 180 * np.pi
        )  # [rad, rad, rad/s, rad/s]
        max_init_state = (
            jnp.array([5.0, 185.0, 0.1, 0.1]) / 180 * np.pi
        )  # [rad, rad, rad/s, rad/s]
        init_state_distribution = UniformDistribution(min_init_state, max_init_state)
        return init_state_distribution.sample(rng_key, shape)

    def make_domain_param_dict(self, params: Array) -> Dict:
        """
        Update domain parameters with the given params.
        :param params:
        :return:
        """
        domain_param_dict = self.ground_truth_params_dict()
        domain_param_dict.update(
            {key: params[i] for i, key in enumerate(self.param_names())}
        )
        return domain_param_dict

    @staticmethod
    def calc_constants(domain_param):
        mass_rot_pole = domain_param["mass_rot_pole"]
        mass_pend_pole = domain_param["mass_pend_pole"]
        length_rot_pole = domain_param["length_rot_pole"]
        length_pend_pole = domain_param["length_pend_pole"]
        gravity_const = domain_param["gravity_const"]

        # Moments of inertia
        Jr = (
            mass_rot_pole * (length_rot_pole**2) / 12
        )  # inertia about COM of the rotary pole [kg*m^2]
        Jp = (
            mass_pend_pole * (length_pend_pole**2) / 12
        )  # inertia about COM of the pendulum pole [kg*m^2]

        # Constants for equations of motion
        c = jnp.stack(
            [
                Jr + mass_pend_pole * length_rot_pole**2,
                0.25 * mass_pend_pole * length_pend_pole**2,
                0.5 * mass_pend_pole * length_pend_pole * length_rot_pole,
                Jp + 0.25 * mass_pend_pole * length_pend_pole**2,
                0.5 * mass_pend_pole * length_pend_pole * gravity_const,
            ],
            axis=-1,
        )
        return c

    @staticmethod
    def observation(state):
        return jnp.stack(
            [
                jnp.sin(state[..., 0]),
                jnp.cos(state[..., 0]),
                jnp.sin(state[..., 1]),
                jnp.cos(state[..., 1]),
                state[..., 2],
                state[..., 3],
            ],
            axis=-1,
        )

    def get_prior(self):
        param_support = support_dict_to_array(self.param_support(), self.param_names())
        return UniformDistribution(min_val=param_support[0], max_val=param_support[1])

    def param_names(self) -> List[str]:
        return [
            "gravity_const",  # gravity [m/s**2]
            # "motor_resistance",  # motor resistance [Ohm]
            # "motor_back_emf",  # motor back-emf constant [V*s/rad]
            "mass_rot_pole",  # rotary arm mass [kg]
            "length_rot_pole",  # rotary arm length [m]
            # "damping_rot_pole",  # rotary arm viscous damping [N*m*s/rad]
            "mass_pend_pole",  # pendulum link mass [kg]
            "length_pend_pole",  # pendulum link length [m]
            # "damping_pend_pole",  # pendulum link viscous damping [N*m*s/rad]
            # "voltage_thold_neg",  # min. voltage in negative direction [V]
            # "voltage_thold_pos",  # min. voltage in positive direction [V]
        ]

    @staticmethod
    def ground_truth_params_dict() -> Dict:
        return {
            "gravity_const": 9.81,  # gravity [m/s**2]
            "motor_resistance": 8.4,  # motor resistance [Ohm]
            "motor_back_emf": 0.042,  # motor back-emf constant [V*s/rad]
            "mass_rot_pole": 0.095,  # rotary arm mass [kg]
            "length_rot_pole": 0.085,  # rotary arm length [m]
            "damping_rot_pole": 5e-6,  # rotary arm viscous damping [N*m*s/rad]
            "mass_pend_pole": 0.024,  # pendulum link mass [kg]
            "length_pend_pole": 0.129,  # pendulum link length [m]
            "damping_pend_pole": 1e-6,  # pendulum link viscous damping [N*m*s/rad]
            "voltage_thold_neg": 0,  # min. voltage in negative direction [V]
            "voltage_thold_pos": 0,  # min. voltage in positive direction [V]
        }

    def param_support(self) -> Dict:
        param_support = {
            "gravity_const": (9.0, 11.0),  # gravity [m/s**2]
            "motor_resistance": (7.0, 9.0),  # motor resistance [Ohm]
            "motor_back_emf": (0.0, 1.0),  # motor back-emf constant [V*s/rad]
            "mass_rot_pole": (0.08, 0.1),  # rotary arm mass [kg]
            "length_rot_pole": (0.08, 0.09),  # rotary arm length [m]
            "damping_rot_pole": (
                0.0,
                1e-2,
            ),  # rotary arm viscous damping [N*m*s/rad]
            "mass_pend_pole": (0.02, 0.03),  # pendulum link mass [kg]
            "length_pend_pole": (0.12, 0.135),  # pendulum link length [m]
            "damping_pend_pole": (
                0.0,
                1e-2,
            ),  # pendulum link viscous damping [N*m*s/rad]
            "voltage_thold_neg": (
                0.0,
                0.1,
            ),  # min. voltage in negative direction [V]
            "voltage_thold_pos": (
                0.0,
                0.1,
            ),  # min. voltage in positive direction [V])
        }
        return {key: param_support[key] for key in self.param_names()}

    def ground_truth_parameters(self) -> np.ndarray:
        return np.array(
            [self.ground_truth_params_dict()[param] for param in self.param_names()]
        )

    @property
    def param_dim(self) -> int:
        return len(self.param_support())

    @property
    def data_dim(self) -> int:
        return 6

    def task_specific_plots(self) -> List:
        plots = super().task_specific_plots()
        return plots + [
            TrajectoryFigure(self.data_dim),
        ]
