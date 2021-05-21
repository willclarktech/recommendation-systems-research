import base64
import json
from typing import Any

import requests
import torch as T
from torch import nn
from websocket import create_connection  # type: ignore

from syft import serialize  # type: ignore
from syft.federated.model_serialization import (  # type: ignore
    deserialize_model_params,
    wrap_model_params,
)


def prettify(json_data: object) -> str:
    return json.dumps(json_data, indent=2).replace("\\n", "\n")


def set_params(model: nn.Module, params: list[T.Tensor]) -> None:
    for p, p_new in zip(model.parameters(), params):
        p.data = p_new.detach().clone().data


def calculate_diff(
    original_params: list[T.Tensor], trained_params: list[T.Tensor]
) -> list[T.Tensor]:
    return [old - new for old, new in zip(original_params, trained_params)]


def send_ws_message(grid_address: str, data: object) -> Any:
    ws = create_connection("ws://" + grid_address)
    ws.send(json.dumps(data))
    message = ws.recv()
    return json.loads(message)


def get_model_params(
    grid_address: str, worker_id: str, request_key: str, model_id: str
) -> list[T.Tensor]:
    get_params = {
        "worker_id": worker_id,
        "request_key": request_key,
        "model_id": model_id,
    }
    response = requests.get(
        f"http://{grid_address}/model-centric/get-model", get_params
    )
    return deserialize_model_params(response.content)


def retrieve_model_params(grid_address: str, name: str, version: str) -> list[T.Tensor]:
    get_params = {
        "name": name,
        "version": version,
        "checkpoint": "latest",
    }

    response = requests.get(
        f"http://{grid_address}/model-centric/retrieve-model", get_params
    )
    return deserialize_model_params(response.content)


def send_auth_request(grid_address: str, name: str, version: str) -> Any:
    message = {
        "type": "model-centric/authenticate",
        "data": {
            "model_name": name,
            "model_version": version,
        },
    }
    return send_ws_message(grid_address, message)


def send_cycle_request(
    grid_address: str, name: str, version: str, worker_id: str
) -> Any:
    message = {
        "type": "model-centric/cycle-request",
        "data": {
            "worker_id": worker_id,
            "model": name,
            "version": version,
            "ping": 1,
            "download": 10000,
            "upload": 10000,
        },
    }
    return send_ws_message(grid_address, message)


def send_diff_report(
    grid_address: str, worker_id: str, request_key: str, diff: list[T.Tensor]
) -> Any:
    serialized_diff = serialize(wrap_model_params(diff)).SerializeToString()
    message = {
        "type": "model-centric/report",
        "data": {
            "worker_id": worker_id,
            "request_key": request_key,
            "diff": base64.b64encode(serialized_diff).decode("ascii"),
        },
    }
    send_ws_message(grid_address, message)
