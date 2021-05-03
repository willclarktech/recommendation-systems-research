NAME = "q-learning"
VERSION = "0.0.0"
GRID_ADDRESS = "localhost:5000"

CLIENT_CONFIG = {
    "name": NAME,
    "version": VERSION,
    "alpha": 0.1,
    "gamma": 0.1,
    "n_train_iterations": 1000,
    "n_test_iterations": 10_000,
    # "max_updates": 1,  # custom syft.js option that limits number of training loops per worker
}

SERVER_CONFIG = {
    "min_workers": 2,
    "max_workers": 2,
    "pool_selection": "random",
    "do_not_reuse_workers_until_cycle": 6,
    "cycle_length": 28800,  # max cycle length in seconds
    "num_cycles": 30,  # max number of cycles
    "max_diffs": 1,  # number of diffs to collect before avg
    "minimum_upload_speed": 0,
    "minimum_download_speed": 0,
    "iterative_plan": True,  # tells PyGrid that avg plan is executed per diff
}

BLACKJACK_DIMS = (32, 11, 2, 2)
