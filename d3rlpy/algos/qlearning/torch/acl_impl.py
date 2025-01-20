import dataclasses
from typing import Callable

import torch

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    ValueFunction,
    build_gaussian_distribution, NSFFunction,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import TorchMiniBatch, CudaGraphWrapper
from ....types import Shape, TorchObservation
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseCriticLoss,
    DDPGBaseImpl,
    DDPGBaseModules,
)

__all__ = ["ACLImpl", "ACLModules"]


@dataclasses.dataclass(frozen=True)
class ACLModules(DDPGBaseModules):
    policy: NormalPolicy
    value_func: ValueFunction
    nsf_s_func: NSFFunction
    nsf_sa_func: NSFFunction
    nsf_optim: OptimizerWrapper


@dataclasses.dataclass(frozen=True)
class ACLNSFLoss:
    nsf_loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ACLCriticLoss(DDPGBaseCriticLoss):
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class ACLImpl(DDPGBaseImpl):
    _modules: ACLModules
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _compute_nsf_grad: Callable[[TorchMiniBatch], ACLNSFLoss]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: ACLModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            compiled=compiled,
            device=device,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight

        self._compute_nsf_grad = self.compute_nsf_grad

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        metrics.update(self.update_nsf(batch))
        self.update_critic_target()
        return metrics

    def update_nsf(self, batch: TorchMiniBatch) -> dict[str, float]:
        loss = self._compute_nsf_grad(batch)
        self._modules.nsf_optim.step()
        return asdict_as_float(loss)

    def compute_nsf_grad(self, batch: TorchMiniBatch) -> ACLNSFLoss:
        self._modules.nsf_optim.zero_grad()
        loss = self.compute_nsf_loss(batch)
        loss.nsf_loss.backward()
        return loss

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> ACLCriticLoss:
        q_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        v_loss = self.compute_value_loss(batch)
        return ACLCriticLoss(
            critic_loss=q_loss + v_loss,
            q_loss=q_loss,
            v_loss=v_loss,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations)

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        # compute log probability
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        return DDPGBaseActorLoss(-(weight * log_probs).mean())

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._modules.value_func(batch.observations)
        adv = q_t - v_t
        cond_as_density = self.get_density_values(batch)
        expectile = 0.5 + (self._expectile - 0.5) * torch.sigmoid(cond_as_density)
        # return (self._weight_temp * adv).exp().clamp(max=self._max_weight)
        return (expectile - (adv < 0.0).float()).abs().detach()

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._modules.value_func(batch.observations)
        cond_as_density = self.get_density_values(batch)
        expectile = 0.5 + (self._expectile - 0.5) * torch.sigmoid(cond_as_density)
        diff = q_t.detach() - v_t
        weight = (expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    @torch.compiler.disable
    def get_density_values(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            nsf_s = self._modules.nsf_s_func(batch.observations)
            nsf_sa = self._modules.nsf_sa_func(torch.cat([batch.observations, batch.actions], dim=-1))
        return nsf_sa - nsf_s

    @torch.compiler.disable
    def compute_nsf_loss(self, batch: TorchMiniBatch) -> ACLNSFLoss:
        nsf_s = self._modules.nsf_s_func(batch.observations)
        nsf_sa = self._modules.nsf_sa_func(torch.cat([batch.observations, batch.actions], dim=-1))
        nsf_s_loss = -nsf_s.mean()
        nsf_sa_loss = -nsf_sa.mean()
        return ACLNSFLoss(nsf_loss=nsf_s_loss + nsf_sa_loss)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()
