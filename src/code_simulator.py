import os
import numpy as np
import qutip as qt
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from multiprocess import Lock
from . import code, noise, recovery

def make_wigner_plots_for(code: code.Code, save_path: Optional[str] = "") -> None:
	x_bounds = (-8, 8)
	y_bounds = (-8, 8)
	x_samples = 600
	y_samples = 600

	zero_encoding_wigner = qt.wigner(qt.ket2dm(code.zero_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	one_encoding_wigner = qt.wigner(qt.ket2dm(code.one_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	plus_encoding_wigner = qt.wigner(qt.ket2dm(code.plus_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	minus_encoding_wigner = qt.wigner(qt.ket2dm(code.minus_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))

	all_wigner_values = np.concatenate((zero_encoding_wigner, one_encoding_wigner, plus_encoding_wigner, minus_encoding_wigner))
	most_extreme_wigner_value_magnitude = np.max(np.abs(all_wigner_values))

	cmap = plt.get_cmap("RdBu")
	normalizer = matplotlib.colors.Normalize(-most_extreme_wigner_value_magnitude, most_extreme_wigner_value_magnitude)

	fig, axes = plt.subplots(2, 2, constrained_layout=True)
	axes[0][0].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), zero_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	axes[0][1].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), one_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	axes[1][0].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), plus_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	axes[1][1].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), minus_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=axes.ravel().tolist())

	for axis in axes.flat:
		axis.set_aspect("equal")

	plt.suptitle(f"{code.name} Wigner Function Plots")
	axes[0][0].set_title("Zero Encoding")
	axes[0][1].set_title("One Encoding")
	axes[1][0].set_title("Plus Encoding")
	axes[1][1].set_title("Minus Encoding")

	if save_path is None:
		return

	if save_path == "":
		if not os.path.exists(f"data/code/{code.name}/"):
			os.makedirs(f"data/code/{code.name}")
		plt.savefig(f"data/code/{code.name}/wigner.png")
	else:
		plt.savefig(save_path)

def get_known_fidelity_for(code_name: str, is_code_random: bool, noise: noise.Noise, use_optimal_recovery: bool) -> Optional[float]:
	if is_code_random:
		complete_path = f"data/code/random/{code_name},{noise},{use_optimal_recovery}/fidelity.txt"
	else:
		complete_path = f"data/code/{code_name}/fidelity/{noise},{use_optimal_recovery}.txt"

	if os.path.exists(complete_path):
		with open(complete_path) as file:
			return float(file.read())
	else:
		return None

def get_fidelity_of(ec_code: code.Code, noise: noise.Noise, use_optimal_recovery: bool, file_io_mutex: Optional[Lock] = None) -> float:
	assert ec_code.physical_dimension == noise.dimension

	if not ec_code.is_random:
		known_fidelity = get_known_fidelity_for(ec_code.name, False, noise, use_optimal_recovery)
		if known_fidelity is not None:
			return known_fidelity

	recovery_matrix = recovery.get_optimal_recovery_matrix(ec_code, noise) if use_optimal_recovery else ec_code.decoder
	fidelity = qt.average_gate_fidelity(recovery_matrix * noise.matrix * ec_code.encoder)

	directory_path = f"data/code/{ec_code.name}/fidelity/"
	complete_path = f"{directory_path}{noise},{use_optimal_recovery}.txt"

	if file_io_mutex is not None:
		file_io_mutex.acquire()
	if ec_code.is_random:
		best_known_fidelity = get_known_fidelity_for(ec_code.name, True, noise, use_optimal_recovery)
		if best_known_fidelity is None or fidelity > best_known_fidelity:
			directory_path = f"data/code/random/{ec_code.name},{noise},{use_optimal_recovery}/"
			complete_path = f"{directory_path}fidelity.txt"
			code.serialize_random_code_with_conditions(ec_code, noise, use_optimal_recovery)
		else:
			if file_io_mutex is not None:
				file_io_mutex.release()
			return fidelity

	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	with open(complete_path, "w") as file:
		file.write(f"{fidelity}")
	if file_io_mutex is not None:
		file_io_mutex.release()

	return fidelity

def compute_code_similarities(code_one: code.Code, code_two: code.Code) -> Tuple[float, float, float, float]:
	code_one_zero = code_one.zero_encoding.data.toarray()
	code_two_zero = code_two.zero_encoding.data.toarray()
	code_one_one = code_one.one_encoding.data.toarray()
	code_two_one = code_two.one_encoding.data.toarray()
	code_one_plus_encoding, code_one_minus_encoding = code.create_plus_and_minus_encodings_from_zero_and_one_encodings(code_one.zero_encoding, code_one.one_encoding)
	code_two_plus_encoding, code_two_minus_encoding = code.create_plus_and_minus_encodings_from_zero_and_one_encodings(code_two.zero_encoding, code_two.one_encoding)
	code_one_plus = code_one_plus_encoding.data.toarray()
	code_one_minus = code_one_minus_encoding.data.toarray()
	code_two_plus = code_two_plus_encoding.data.toarray()
	code_two_minus = code_two_minus_encoding.data.toarray()

	if code_one.physical_dimension > code_two.physical_dimension:
		code_one_zero, code_two_zero = code_two_zero, code_one_zero
		code_one_one, code_two_one = code_two_one, code_one_one
		code_one_plus, code_two_plus = code_two_plus, code_one_plus
		code_one_minus, code_two_minus = code_two_minus, code_one_minus

	code_two_zero.resize(code_one_zero.shape, refcheck=False)
	code_two_one.resize(code_one_one.shape, refcheck=False)
	code_two_plus.resize(code_one_plus.shape, refcheck=False)
	code_two_minus.resize(code_one_minus.shape, refcheck=False)

	return (
		np.abs(np.vdot(code_one_zero, code_two_zero)) ** 2,
		np.abs(np.vdot(code_one_one, code_two_one)) ** 2,
		np.abs(np.vdot(code_one_plus, code_two_plus)) ** 2,
		np.abs(np.vdot(code_one_minus, code_two_minus)) ** 2
	)
