import os
import numpy as np
import qutip as qt
from typing import List, Tuple

class Noise:

	def __init__(self, dimension: int, noise_names_and_amounts: List[Tuple[str, float]], number_of_terms: int = 50):
		assert all(name in noise_name_to_constructor_function for name, _ in noise_names_and_amounts)
		self.dimension = dimension
		self.description = noise_names_and_amounts
		self.number_of_terms = number_of_terms
		self.matrix = qt.kraus_to_super([qt.Qobj(np.identity(dimension))])
		for noise_name, noise_amount in reversed(noise_names_and_amounts):
			self.matrix *= noise_name_to_constructor_function[noise_name](dimension, noise_amount, number_of_terms)

	def __repr__(self):
		return "-".join([f"{noise_name}-{noise_amount}" for noise_name, noise_amount in self.description])

def get_loss_noise_matrix(dimension: int, noise_amount: float, number_of_terms: int = 50) -> qt.Qobj:
	if not os.path.exists(f"data/noise/"):
		os.makedirs(f"data/noise/")
	path = f"data/noise/loss-{dimension},{noise_amount},{number_of_terms}"
	if os.path.exists(f"{path}.qu"):
		return qt.qload(path)
	lowering_operator = qt.destroy(dimension)
	x = np.exp(-noise_amount)
	number_power_operator = qt.Qobj(np.diag([np.sqrt(x) ** i for i in range(dimension)]))
	c = 1
	kraus_list = [number_power_operator]
	for l in range(1, number_of_terms):
		c *= (1 - x) / l
		kraus_list.append(np.sqrt(c) * number_power_operator * lowering_operator ** l)
	noise_matrix = qt.kraus_to_super(kraus_list)
	qt.qsave(noise_matrix, path)
	return noise_matrix

def get_dephasing_noise_matrix(dimension: int, noise_amount: float, number_of_terms: int = 50) -> qt.Qobj:
	if not os.path.exists(f"data/noise/"):
		os.makedirs(f"data/noise/")
	path = f"data/noise/dephasing-{dimension},{noise_amount},{number_of_terms}"
	if os.path.exists(f"{path}.qu"):
		return qt.qload(path)
	c = 1
	kraus_list = []
	for l in range(number_of_terms):
		c *= noise_amount / l if l != 0 else 1
		kraus_list.append(qt.Qobj(np.diag(np.sqrt(c) * np.exp(-0.5 * noise_amount * np.arange(dimension) ** 2) * np.arange(dimension) ** l)))
	noise_matrix = qt.kraus_to_super(kraus_list)
	qt.qsave(noise_matrix, path)
	return noise_matrix

noise_name_to_constructor_function = {
	"loss": get_loss_noise_matrix,
	"dephasing": get_dephasing_noise_matrix,
}
