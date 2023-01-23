import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from . import code

def make_wigner_plots_for(code: code.Code):
	x_bounds = (-8, 8)
	y_bounds = (-8, 8)
	x_samples = 600
	y_samples = 600

	zero_encoding_wigner = qt.wigner(qt.ket2dm(code.zero_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	one_encoding_wigner = qt.wigner(qt.ket2dm(code.one_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	plus_encoding_wigner = qt.wigner(qt.ket2dm(code.plus_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))
	minus_encoding_wigner = qt.wigner(qt.ket2dm(code.minus_encoding), np.linspace(*x_bounds, x_samples), np.linspace(*y_bounds, y_samples))

	if not os.path.exists(f"data/{code.name}/"):
		os.makedirs(f"data/{code.name}")

	fig, axes = plt.subplots(2, 2, constrained_layout=True)
	axes[0][0].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), zero_encoding_wigner, 100, cmap=plt.cm.RdBu)
	axes[0][1].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), one_encoding_wigner, 100, cmap=plt.cm.RdBu)
	axes[1][0].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), plus_encoding_wigner, 100, cmap=plt.cm.RdBu)
	contour = axes[1][1].contourf(np.linspace(*x_bounds, x_samples) / np.sqrt(2), np.linspace(*x_bounds, x_samples) / np.sqrt(2), minus_encoding_wigner, 100, cmap=plt.cm.RdBu)
	fig.colorbar(contour, ax=axes.ravel().tolist())

	for axis in axes.flat:
		axis.set_aspect("equal")

	plt.suptitle(f"{code.name} Wigner Function Plots")
	axes[0][0].set_title("Zero Encoding")
	axes[0][1].set_title("One Encoding")
	axes[1][0].set_title("Plus Encoding")
	axes[1][1].set_title("Minus Encoding")

	plt.savefig(f"data/{code.name}/wigner.png")

make_wigner_plots_for(code.BinomialCode(3, 10, 36))
