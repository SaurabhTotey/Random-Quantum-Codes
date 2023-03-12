import os
import numpy as np
import qutip as qt
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from . import code, noise, recovery

def make_wigner_plots_for(code: code.Code) -> None:
	x_bounds = (-8, 8)
	y_bounds = (-8, 8)
	x_samples = 600
	y_samples = 600

	zero_encoding_wigner = qt.wigner(qt.ket2dm(code.zero_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	one_encoding_wigner = qt.wigner(qt.ket2dm(code.one_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	plus_encoding_wigner = qt.wigner(qt.ket2dm(code.plus_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	minus_encoding_wigner = qt.wigner(qt.ket2dm(code.minus_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))

	all_wigner_values = np.concatenate((zero_encoding_wigner, one_encoding_wigner, plus_encoding_wigner, minus_encoding_wigner))
	smallest_wigner_value = np.min(all_wigner_values)
	largest_wigner_value = np.max(all_wigner_values)

	cmap = plt.get_cmap("RdBu")
	normalizer = matplotlib.colors.Normalize(smallest_wigner_value, largest_wigner_value)

	fig, axes = plt.subplots(2, 2, constrained_layout=True)
	axes[0][0].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), zero_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	axes[0][1].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), one_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	axes[1][0].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), plus_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	axes[1][1].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), minus_encoding_wigner, 100, cmap=cmap, norm=normalizer)
	fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=axes.ravel().tolist())

	for axis in axes.flat:
		axis.set_aspect("equal")

	plt.suptitle(f"{code.family_name} Wigner Function Plots")
	axes[0][0].set_title("Zero Encoding")
	axes[0][1].set_title("One Encoding")
	axes[1][0].set_title("Plus Encoding")
	axes[1][1].set_title("Minus Encoding")

	if not os.path.exists(f"data/code/{code.family_name}/"):
		os.makedirs(f"data/code/{code.family_name}")
	plt.savefig(f"data/code/{code.family_name}/wigner.png")

def get_fidelity_of_code_under_loss_noise(code: code.Code, loss_noise_amount: float, use_optimal_recovery: bool) -> float:
	directory_path = f"data/code/{code.family_name}/"
	complete_path = f"{directory_path}fidelity-{loss_noise_amount},{use_optimal_recovery}.txt"
	if not code.is_random and os.path.exists(complete_path):
		with open(complete_path) as file:
			return float(file.read())
	noise_matrix = noise.get_loss_noise_matrix(code.physical_dimension, loss_noise_amount)
	recovery_matrix = recovery.get_optimal_recovery_matrix_for_loss_channel(code, loss_noise_amount) if use_optimal_recovery else code.decoder
	fidelity = qt.average_gate_fidelity(recovery_matrix * noise_matrix * code.encoder)
	if not code.is_random:
		if not os.path.exists(directory_path):
			os.makedirs(directory_path)
		with open(complete_path, "w") as file:
			file.write(f"{fidelity}")
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
		np.vdot(code_one_zero, code_two_zero),
		np.vdot(code_one_one, code_two_one),
		np.vdot(code_one_plus, code_two_plus),
		np.vdot(code_one_minus, code_two_minus)
	)
