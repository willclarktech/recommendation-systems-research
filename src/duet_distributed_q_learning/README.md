# Distributed Q-Learning

This directory demonstrates distributed Q-Learning using PySyft and Duet (all running on a local network). It is a step towards federated Q-Learning.

## Setting up a PySyft network

You could use OpenGrid, which is the default network used by Duet to connect data owners with data scientists. It is a free public network, however I found it to be rather unreliable. Alternatively you can run `syft-network` to spin up a local network running by default on `http://0.0.0.0:5000/`, which is the approach taken here.

## Start the data scientist and data owner scripts

You can run the data scientist script like this:

```sh
python data_scientist.py <number of data owners>
```

This will set up a training loop on the data scientist side which will connect to the data owners when they are available, initialize a Q-table, and repeatedly distribute this Q-table and aggregate the training results from the data owners. The data owners actually do the training in an OpenAI Blackjack gym environment, waiting for the updated Q-table from the data scientist before beginning the next training epoch.

You can run the data owner script like this (run the same command in a separate terminal for each data owner):

```sh
python data_owner.py
```

Before training, the data scientist will provide a baseline perfomance measure, and after training is complete the data scientist will repeat the test with the trained Q-table.

The data owner script must be exited manually because it runs forever in order to allow the data scientist script as much time as it needs to access the Duet store.

All shared code can be found in `shared.py`.

## To-do

-   [ ] Filter logging to exclude `CRITICAL` log messages which are just the result of polling for objects before they exist.
-   [ ] Allow more options to be set via the CLI.
