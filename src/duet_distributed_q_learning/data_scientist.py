import asyncio
import sys

import syft as sy
import torch as T

from shared import (
    NETWORK_URL,
    N_EPOCHS,
    OWNER_TAG,
    SCIENTIST_TAG,
    run_epoch,
    wait_for_remote_object,
)

USAGE_MESSAGE = "usage: python data_scientist.py <n_data_owners>"


async def train(duets, initial_table):
    initial_table.tag(SCIENTIST_TAG)
    initial_table.describe("A Q-table with initial values")

    for duet in duets:
        initial_table.send(duet)

    remote_diffs = [await wait_for_remote_object(duet, OWNER_TAG) for duet in duets]
    local_diffs = [
        remote_diff.get(reason="Needed for update", request_block=True)
        for remote_diff in remote_diffs
    ]

    return initial_table + T.mean(T.stack(local_diffs), dim=0)


async def main(n_data_owners: int) -> None:
    duets = [
        sy.join_duet(network_url=NETWORK_URL, loopback=True)
        for _ in range(n_data_owners)
    ]

    n_games_per_epoch = 10_000

    q_table = T.zeros((32, 11, 2, 2))
    _, pre_rets = run_epoch(n_games_per_epoch, q_table, train=False)
    print(f"Initial agent - Average return: {sum(pre_rets) / len(pre_rets):.4f}")

    for _ in range(N_EPOCHS):
        q_table = await train(duets, q_table)

    _, post_rets = run_epoch(n_games_per_epoch, q_table, train=False)
    print(f"Trained agent - Average return: {sum(post_rets) / len(post_rets):.4f}")


if __name__ == "__main__":
    try:
        n_data_owners = int(sys.argv[1])
    except IndexError as e:
        print(USAGE_MESSAGE)
        sys.exit(1)
    except ValueError as e:
        print(USAGE_MESSAGE)
        sys.exit(1)

    asyncio.run(main(n_data_owners))
    print("Done")
