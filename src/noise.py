import numpy as np
import qutip as qt

def make_loss_noise_matrix(dimension: int, noise_amount: float, number_of_terms: int = 50) -> qt.Qobj:
	lowering_operator = qt.destroy(dimension)
	x = np.exp(-noise_amount)
	number_power_operator = qt.Qobj(np.diag([np.sqrt(x) ** i for i in range(dimension)]))
	c = 1
	kraus_list = [number_power_operator]
	for l in range(1, number_of_terms):
		c *= (1 - x) / l
		kraus_list.append(np.sqrt(c) * number_power_operator * lowering_operator ** l)
	return qt.kraus_to_super(kraus_list)
