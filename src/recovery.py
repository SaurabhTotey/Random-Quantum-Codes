import os
import cvxpy as cp
import numpy as np
import qutip as qt
import scipy
from . import code, noise

def get_optimal_recovery_matrix(code: code.Code, noise: noise.Noise) -> qt.Qobj:
	directory_path = f"data/code/{code.name}/recovery/"
	complete_path = f"{directory_path}optimal-{noise}"
	if not code.is_random:
		if os.path.exists(f"{complete_path}.qu"):
			return qt.qload(complete_path)

	total_dimension = code.physical_dimension * code.code_dimension
	choi_matrix = qt.super_to_choi((1 / code.code_dimension ** 2) * (noise.matrix * code.encoder).dag())
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
		if not os.path.exists(directory_path):
			os.makedirs(directory_path)
		qt.qsave(recovery_matrix, complete_path)
	return recovery_matrix

def save_optimal_recovery_matrix_for_random_code(code: code.Code, noise: noise.Noise, optimal_recovery_matrix: qt.Qobj) -> None:
	directory_path = f"data/code/random/{code.name},{noise},True/recovery/"
	complete_path = f"{directory_path}optimal"
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	qt.qsave(optimal_recovery_matrix, complete_path)
