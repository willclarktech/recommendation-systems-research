from typing import Any

import torch as T

import syft as sy
from syft import make_plan
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.lib.python.int import Int
from syft.lib.python.list import List

from config import (
    CLIENT_CONFIG,
    GRID_ADDRESS,
    HIDDEN_WIDTH,
    INPUT_WIDTH,
    OUTPUT_WIDTH,
    SERVER_CONFIG,
)
from utils import prettify


class DQNAgent(sy.Module):
    def __init__(
        self,
        torch_ref: Any,
        input_width,
        output_width,
        hidden_width,
        initial_epsilon=1.0,
    ) -> None:
        super().__init__(torch_ref=torch_ref)
        nn = torch_ref.nn
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, output_width),
        )
        # HACK: We can’t transfer parameters except via a PyTorch Module
        self.epsilon = nn.Linear(1, 1, bias=False)
        for p in self.epsilon.parameters():
            p.requires_grad = False
        self.epsilon.weight.data = torch_ref.tensor(
            [initial_epsilon], requires_grad=False
        ).data


def main() -> None:
    local_agent = DQNAgent(T, INPUT_WIDTH, OUTPUT_WIDTH, HIDDEN_WIDTH)

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
