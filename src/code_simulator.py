import os
import numpy as np
import qutip as qt
import matplotlib
import matplotlib.pyplot as plt
from . import code, noise

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

	plt.suptitle(f"{code.name} Wigner Function Plots")
	axes[0][0].set_title("Zero Encoding")
	axes[0][1].set_title("One Encoding")
	axes[1][0].set_title("Plus Encoding")
	axes[1][1].set_title("Minus Encoding")

	if not os.path.exists(f"data/{code.name}/"):
		os.makedirs(f"data/{code.name}")
	plt.savefig(f"data/{code.name}/wigner.png")

def get_no_recovery_fidelity_of_code_under_loss_noise(code: code.Code, loss_noise_amount: float) -> float:
	return qt.average_gate_fidelity(noise.make_noise_loss_matrix(loss_noise_amount) * code.encoder)

def get_optimal_recovery_fidelity_of_code_under_loss_noise(code: code.Code, loss_noise_amount: float) -> float:
	noise_matrix = noise.make_noise_loss_matrix(loss_noise_amount)
	return qt.average_gate_fidelity(code.compute_optimal_recovery_matrix_through(noise_matrix) * noise_matrix * code.encoder)
