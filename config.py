import numpy as np

HARDWARE = {
    'Mp': 0.027,
    'Lp': 0.191,
    'lp': 0.153,
    'Jp': 1.10e-4,
    'Mr': 0.028,
    'Lr': 0.0826,
    'Jr': 1.23e-4,
    'Rm': 3.3,
    'kt': 0.02797,
    'km': 0.02797,
    'g': 9.81,
    'Dr': 0.005,
    'Dp': 0.001,
}


CONSTRAINTS = {
    'max_voltage': 12.0,
    'dt': 0.01,
    'max_steps': 2500,
}
