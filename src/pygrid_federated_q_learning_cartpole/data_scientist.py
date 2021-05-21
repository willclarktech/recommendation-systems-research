from math import prod
from typing import Any

import torch as T

import syft as sy
from syft import make_plan
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.lib.python.int import Int
from syft.lib.python.list import List

from config import CARTPOLE_DIMS, CLIENT_CONFIG, GRID_ADDRESS, SERVER_CONFIG
from utils import prettify

# Hacky way to make a Q-table which PyGrid will host.
# 1. We need the model itself to be a sy.Module.
# 2. We need its parameters to be added via torch.nn.Module instances.
# 3. Those instances must be instantiated via torch.nn.Module subclasses imported from torch.
# Solution: store the Q-table tensor as the weights in a Linear module and extract/reshape on the client after downloading the model.
# The bias represents the epsilon value
class QLearningAgent(sy.Module):
    def __init__(
        self,
        torch_ref: Any,
        dims: tuple[int, ...],
        initial_epsilon=1.0,
    ) -> None:
        super().__init__(torch_ref=torch_ref)
        n_weights = prod(dims)
        self.network = torch_ref.nn.Linear(n_weights, 1, bias=True)
        for p in self.parameters():
            p.requires_grad = False
        torch_ref.nn.init.zeros_(self.network.weight)
        self.network.bias.data = torch_ref.tensor(
            [initial_epsilon], requires_grad=False
        ).data


def main() -> None:
    local_agent = QLearningAgent(T, CARTPOLE_DIMS)

    # Hack: we need a client plan so PyGrid will allow our client to download the model
    # We can't use a training plan right now because the Blackjack env isn't supported yet
    @make_plan
    def nothing_plan():
        pass

    @make_plan
    def averaging_plan(
        # Average of diffs, not parameters
        current_average=List(local_agent.parameters()),
        next_diff=List(local_agent.parameters()),
        num=Int(0),
    ):
        return [
            (current_param * num + diff_param) / (num + 1)
            for current_param, diff_param in zip(current_average, next_diff)
        ]

    grid = ModelCentricFLClient(address=GRID_ADDRESS, secure=False)
    grid.connect()

    response = grid.host_federated_training(
        model=local_agent,
        client_plans={"nothing_plan": nothing_plan},
        client_protocols={},
        server_averaging_plan=averaging_plan,
        client_config=CLIENT_CONFIG,
        server_config=SERVER_CONFIG,
    )

    print(f"Host response: {prettify(response)}")


if __name__ == "__main__":
    main()
