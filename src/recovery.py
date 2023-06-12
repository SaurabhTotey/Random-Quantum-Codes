import os
import cvxpy as cp
import numpy as np
import qutip as qt
import scipy
from . import code, noise

def get_optimal_recovery_matrix_for_noise_channel(code: code.Code, noise: noise.Noise) -> qt.Qobj:
	if not code.is_random:
		if not os.path.exists(f"data/recovery/"):
			os.makedirs(f"data/recovery/")
		path = f"data/recovery/optimal,code-{code.family_name},{noise}"
		if os.path.exists(f"{path}.qu"):
			return qt.qload(path)
	noise_matrix = noise.matrix
	total_dimension = code.physical_dimension * code.code_dimension
	choi_matrix = qt.super_to_choi((1 / code.code_dimension ** 2) * (noise_matrix * code.encoder).dag())
	sdp_solution = cp.Variable((total_dimension, total_dimension), complex=True)
	objective = cp.Maximize(cp.real(cp.trace(sdp_solution @ choi_matrix)))
	cp.Problem(objective, [
		cp.partial_trace(sdp_solution, [code.physical_dimension, code.code_dimension], 1) == np.identity(code.physical_dimension),
		sdp_solution >> 0,
		sdp_solution.H == sdp_solution,
	]).solve()
	recovery_matrix = qt.choi_to_super(
		qt.Qobj(
			scipy.sparse.csr_matrix(sdp_solution.value),
			dims=[[[code.physical_dimension], [code.code_dimension]], [[code.physical_dimension], [code.code_dimension]]],
			superrep="choi",
		)
	)
	if not code.is_random:
		qt.qsave(recovery_matrix, path)
	return recovery_matrix
