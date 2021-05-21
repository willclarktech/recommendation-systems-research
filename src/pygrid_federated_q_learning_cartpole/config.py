NAME = "q-learning"
VERSION = "0.0.0"
GRID_ADDRESS = "localhost:5000"

CLIENT_CONFIG = {
    "name": NAME,
    "version": VERSION,
    "alpha": 0.1,
    "gamma": 1.0,
    "min_epsilon": 0.1,
    "epsilon_reduction": 0.001,
    "n_train_iterations": 300,
    "n_test_iterations": 100,
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

CARTPOLE_DIMS = (1, 1, 12, 6, 2)
