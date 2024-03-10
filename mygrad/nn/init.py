# nn/init.py
import numpy as np


def get_gain_by_activation(activation_name, param=None):
    if (activation_name == "Identity") or (activation_name == "Sigmoid") or (activation_name == "Conv"):
        return 1.0
    elif activation_name == "Tanh":
        return 5 / 3
    elif activation_name == "ReLU":
        return np.sqrt(2)
    elif activation_name == "LeakyReLU":
        param = param if param is not None else 1e-2
        return np.sqrt(2 / (1 + param**2))
    else:
        print(f"Unknown activation: f{activation_name}. Gain is set to 1.0")
        return 1.0


def xavier_normal_initialization(v, gain=1.0, activation=None, param=None):
    if activation is not None:
        gain = get_gain_by_activation(activation, param)

    n_in = v.shape[0]
    n_out = 1
    if v.ndim > 1:
        n_out = v.shape[1]

    std = gain * np.sqrt(2 / (n_in + n_out))
    v.set_data(std * np.random.randn(n_in, n_out))


def xavier_uniform_initialization(v, gain=1.0, activation=None, param=None):
    if activation is not None:
        gain = get_gain_by_activation(activation, param)

    n_in = v.shape[0]
    n_out = 1
    if v.ndim > 1:
        n_out = v.shape[1]

    x = gain * np.sqrt(6 / (n_in + n_out))
    v.set_data(np.random.uniform(-x, x, (n_in, n_out)))


def he_normal_initialization(v, gain=1.0, mode="n_in", activation=None, param=None):
    if activation is not None:
        gain = get_gain_by_activation(activation, param)

    n_in = v.shape[0]
    n_out = 1
    if v.ndim > 1:
        n_out = v.shape[1]

    n = None
    if mode == "n_in":
        n = n_in
    if mode == "n_out":
        n = n_out
    else:
        print(f"Invalid mode: {mode}. Using n_in")
        n = n_in

    std = gain * np.sqrt(1 / n)
    v.set_data(std * np.random.randn(n_in, n_out))


def he_uniform_initialization(v, gain=1.0, mode="n_in", activation=None, param=None):
    if activation is not None:
        gain = get_gain_by_activation(activation, param)

    n_in = v.shape[0]
    n_out = 1
    if v.ndim > 1:
        n_out = v.shape[1]

    n = None
    if mode == "n_in":
        n = n_in
    if mode == "n_out":
        n = n_out
    else:
        print(f"Invalid mode: {mode}. Using n_in")
        n = n_in

    x = gain * np.sqrt(3 / n)
    v.set_data(np.random.uniform(-x, x, size=(n_in, n_out)))
