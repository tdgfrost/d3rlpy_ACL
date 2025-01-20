import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_value_function,
    create_nsf_function,
)
from ...models.encoders import EncoderFactory, make_encoder_field, NSFEncoderFactory
from ...models.q_functions import MeanQFunctionFactory
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.acl_impl import ACLImpl, ACLModules

__all__ = ["ACLConfig", "ACL"]


@dataclasses.dataclass()
class ACLConfig(LearnableConfig):
    r"""Advantage-Consistent Learning algorithm.

    IQL is the offline RL algorithm that avoids ever querying values of unseen
    actions while still being able to perform multi-step dynamic programming
    updates. ACL self-regularises this using state and state-action densities.

    There are three functions to train in IQL and two additional functions for ACL. First the state-value function
    is trained via expectile regression.

    .. math::

        L_V(\psi) = \mathbb{E}_{(s, a) \sim D}
            [L_2^\tau (Q_\theta (s, a) - V_\psi (s))]

    where :math:`L_2^\tau (u) = |\tau - \mathbb{1}(u < 0)|u^2`.

    The Q-function is trained with the state-value function to avoid query the
    actions.

    .. math::

        L_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}
            [(r + \gamma V_\psi(s') - Q_\theta(s, a))^2]

    The policy function is trained by using advantage weighted
    regression.

    .. math::

        L_\pi (\phi) = \mathbb{E}_{(s, a) \sim D}
            [\exp(\beta (Q_\theta - V_\psi(s))) \log \pi_\phi(a|s)]

    The density ratio for state and state-action pairs is trained using Neural Spline Flows (NSF).
    We then take f(a|s) = f(s, a) / f(s) to get the density ratio for actions conditioned on the state.

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the value function.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        expectile (float): Expectile value for value function training.
        weight_temp (float): Inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): Maximum advantage weight value to clip.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    nsf_learning_rate: float = 1e-3  # Added
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    nsf_optim_factory: OptimizerFactory = make_optimizer_field()  # Added
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    nsf_encoder_factory: EncoderFactory = NSFEncoderFactory()  # Added
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "ACL":
        return ACL(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "acl"


class ACL(QLearningAlgoBase[ACLImpl, ACLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        nsf_s_function = create_nsf_function(
            observation_shape[0],
            self._config.nsf_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        nsf_sa_function = create_nsf_function(
            observation_shape[0] + action_size,
            self._config.nsf_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        q_func_params = list(q_funcs.named_modules())
        v_func_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_func_params + v_func_params,
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
        nsf_s_func_params = list(nsf_s_function.named_modules())
        nsf_sa_func_params = list(nsf_sa_function.named_modules())
        nsf_optim = self._config.nsf_optim_factory.create(
            nsf_s_func_params + nsf_sa_func_params,
            lr=self._config.nsf_learning_rate,
            compiled=self.compiled,
        )

        modules = ACLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            nsf_s_func=nsf_s_function,
            nsf_sa_func=nsf_sa_function,
            nsf_optim=nsf_optim,
        )

        self._impl = ACLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(ACLConfig)
