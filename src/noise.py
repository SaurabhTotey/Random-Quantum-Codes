import os
import numpy as np
import qutip as qt

def get_loss_noise_matrix(dimension: int, noise_amount: float, number_of_terms: int = 50) -> qt.Qobj:
	if not os.path.exists(f"data/channels/"):
		os.makedirs(f"data/channels/")
	path = f"data/channels/loss,{noise_amount}.txt"
	if os.path.exists(path):
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
