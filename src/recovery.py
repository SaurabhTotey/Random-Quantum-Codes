import cvxpy as cp
import numpy as np
import qutip as qt
import scipy
from . import code, noise

def compute_optimal_recovery_for_loss_channel(code: code.Code, noise_amount: float) -> qt.Qobj:
	# TODO: save results if possible, and get results from file if code is not random
	loss_noise_matrix = noise.get_loss_noise_matrix(code.physical_dimension, noise_amount)
	total_dimension = code.physical_dimension * code.code_dimension
	choi_matrix = qt.super_to_choi((1 / code.code_dimension ** 2) * (loss_noise_matrix * code.encoder).dag())
	sdp_solution = cp.Variable((total_dimension, total_dimension), complex=True)
	objective = cp.Maximize(cp.real(cp.trace(sdp_solution @ choi_matrix)))
	cp.Problem(objective, [
		cp.partial_trace(sdp_solution, [code.physical_dimension, code.code_dimension], 1) == np.identity(code.physical_dimension),
		sdp_solution >> 0,
		sdp_solution.H == sdp_solution,
	]).solve()
	return qt.choi_to_super(
		qt.Qobj(
			scipy.sparse.csr_matrix(sdp_solution.value),
			dims=[[[code.physical_dimension], [code.code_dimension]], [[code.physical_dimension], [code.code_dimension]]],
			superrep="choi",
		)
	)
