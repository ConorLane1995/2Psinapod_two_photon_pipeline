import numpy as np


def get_cell_x(cell):
    return cell["x"]


def get_cell_y(cell):
    return cell["y"]

def get_best_frequency(cell):
    tuning_curve = cell['tuning_curve']
    max_response_idx = np.argmax(tuning_curve[:,1])
    return tuning_curve[max_response_idx,0]