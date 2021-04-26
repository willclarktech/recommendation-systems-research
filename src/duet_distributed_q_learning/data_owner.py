import asyncio
import syft as sy

from shared import (
    NETWORK_URL,
    N_EPOCHS,
    OWNER_TAG,
    SCIENTIST_TAG,
    run_epoch,
    wait_for_remote_object,
)


async def train(duet, epoch, n_iterations_per_epoch):
    initial_q_table_remote = await wait_for_remote_object(duet, SCIENTIST_TAG)
    initial_q_table = (
        initial_q_table_remote.get()
    )  # request_block=True returns None after timing out

    trained_q_table, rets = run_epoch(
        n_iterations_per_epoch, initial_q_table, train=True
    )

    diff = trained_q_table - initial_q_table
    diff.tag(OWNER_TAG)
    diff.describe(f"Epoch {epoch} diff")
    diff.send(duet, pointable=True)

    return rets


async def main():
    duet = sy.launch_duet(network_url=NETWORK_URL, loopback=True)
    duet.requests.add_handler(action="accept")

    n_iterations_per_epoch = 10_000

    for i in range(N_EPOCHS):
        rets = await train(duet, i, n_iterations_per_epoch)
        print(f"Epoch {i+1} - Average return: {sum(rets) / len(rets):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
    print("Done. Looping forever to keep Duet alive...")
    sy.event_loop.loop.run_forever()
