import os
import numpy as np
import qutip as qt
import itertools
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Tuple
import multiprocess
import multiprocess.pool
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

def get_fidelity_of(ec_code: code.Code, noise: noise.Noise, use_optimal_recovery: bool, random_code_file_io_mutex: Optional[multiprocess.Lock] = None) -> float:
	assert ec_code.physical_dimension == noise.dimension
	if random_code_file_io_mutex is not None:
		assert ec_code.is_random

	if not ec_code.is_random:
		known_fidelity = get_known_fidelity_for(ec_code.name, False, noise, use_optimal_recovery)
		if known_fidelity is not None:
			return known_fidelity

	recovery_matrix = recovery.get_optimal_recovery_matrix(ec_code, noise) if use_optimal_recovery else ec_code.decoder
	fidelity = qt.average_gate_fidelity(recovery_matrix * noise.matrix * ec_code.encoder)

	directory_path = f"data/code/{ec_code.name}/fidelity/"
	complete_path = f"{directory_path}{noise},{use_optimal_recovery}.txt"

	if ec_code.is_random:
		if random_code_file_io_mutex is not None:
			random_code_file_io_mutex.acquire()
		best_known_fidelity = get_known_fidelity_for(ec_code.name, True, noise, use_optimal_recovery)
		if best_known_fidelity is None or fidelity > best_known_fidelity:
			directory_path = f"data/code/random/{ec_code.name},{noise},{use_optimal_recovery}/"
			complete_path = f"{directory_path}fidelity.txt"
			code.serialize_random_code_with_conditions(ec_code, noise, use_optimal_recovery)
		else:
			if random_code_file_io_mutex is not None:
				random_code_file_io_mutex.release()
			return fidelity

	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	with open(complete_path, "w") as file:
		file.write(f"{fidelity}")
	if ec_code.is_random and random_code_file_io_mutex is not None:
		random_code_file_io_mutex.release()

	return fidelity

def compute_code_similarities(code_one: code.Code, code_two: code.Code) -> Tuple[float, float, float, float]:
	code_one_zero = code_one.zero_encoding.data.toarray()
	code_two_zero = code_two.zero_encoding.data.toarray()
	code_one_one = code_one.one_encoding.data.toarray()
	code_two_one = code_two.one_encoding.data.toarray()
	code_one_plus = code_one.plus_encoding.data.toarray()
	code_one_minus = code_one.minus_encoding.data.toarray()
	code_two_plus = code_two.plus_encoding.data.toarray()
	code_two_minus = code_two.minus_encoding.data.toarray()

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

def run_parameter_sweep_for_optimal_fidelities(code_parameters: List[List[Any]], noise_parameters: List[List[Any]], make_code_from_parameters: Callable[..., code.Code], make_noise_channel_from_parameters: Callable[..., noise.Noise], number_of_trials_per_parameter_set: Optional[int] = None) -> np.ndarray:
	"""
	This function sweeps over the given parameters and gets fidelities for the given codes under the given noise
	channels. The code_parameters argument is a list of lists, where each sub-list is the list of possible values that a
	specific parameter can take on. For example, if code_parameters = [[0, 1], ["a", "b", "c"]], then the first code
	parameter can either have values of 0 and 1, and the second can take on the value of "a" or "b" or "c". These
	parameters are passed into the make_code_from_parameters argument, so make_code_from_parameters will return Codes
	using parameters (0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), and (1, "c"). The noise_parameters argument
	behaves similarly but should contain parameter values for noise channels. However, the
	make_noise_channel_from_parameters argument doesn't just take in noise parameters. It takes in all the noise
	parameters and code parameters in that order. This is so that the noise channel can adequately act on the
	corresponding code and have the right physical dimension and such. The number_of_trials_per_parameter_set specifies
	how many times to repeat a "trial" with a set of fixed code parameters and noise channel. For each trial, the code
	is regenerated with make_code_from_parameters and its fidelity is found. This is mainly useful for random codes
	since make_code_from_parameters can return a new code even when given the same parameters for random codes. It is
	assumed that non-random codes will always generate the same when given the same parameters, and accordingly, the
	number_of_trials_per_parameter_set value should be None or 1 for non-random codes. The return value is an array of
	fidelities. The shape of the returned array is (len(noise_parameters[0]), len(noise_parameters[1]), ...,
	len(noise_parameters[n]), len(code_parameters[0]), len(code_parameters[1]), ..., len(code_parameters[m])) where n
	is the number of noise parameters and m is the number of code parameters. If number_of_trials_per_parameter_set is
	not None, then there is an additional dimension of size number_of_trials_per_parameter_set for the fidelity of each
	trial. The indices used to access the returned fidelities correspond to the indices of the parameters used in making
	that fidelity.

	TODO: add progress bar for jupyter notebooks
	"""

	all_parameters = noise_parameters + code_parameters

	# Generate all noise channels beforehand so that there is no issue with multiprocessing. This isn't parallelized because it tends to be quick.
	# Additionally, parallelization would be difficult because, while we may assume that unique noise parameters lead to unique noise channels,
	# it isn't necessarily true that that holds when also accounting for code parameters being used to create noise channels. It is possible
	# that different code parameters could still be used in the creation of the same noise channel.
	noise_channels = np.empty(tuple(len(specific_parameter_values) for specific_parameter_values in all_parameters), noise.Noise)
	number_of_indices = np.product(noise_channels.shape)
	for i in range(number_of_indices):
		parameter_indices = np.unravel_index(i, noise_channels.shape)
		noise_channels[parameter_indices] = make_noise_channel_from_parameters(*[all_parameters[j][k] for j, k in enumerate(parameter_indices)])

	fidelities_shape = tuple(len(specific_parameter_values) for specific_parameter_values in all_parameters)
	if number_of_trials_per_parameter_set is not None:
		fidelities_shape = (*fidelities_shape, number_of_trials_per_parameter_set)
	fidelities = np.zeros(fidelities_shape)

	def initialize_pool(lock_instance):
		global parameter_sweep_lock
		parameter_sweep_lock = lock_instance
	with multiprocess.Pool(initializer=initialize_pool, initargs=(multiprocess.Lock(),)) as pool:

		# If codes are only being used once, pre-generate them because we want to ensure there are no multiprocessing issues if they're not random.
		# It is assumed that unique code parameters lead a unique code for non-random codes.
		pregenerated_codes_shape = tuple(len(code_specific_parameter_values) for code_specific_parameter_values in code_parameters)
		if number_of_trials_per_parameter_set is not None:
			pregenerated_codes_shape = (*pregenerated_codes_shape, number_of_trials_per_parameter_set)
		pregenerated_codes = np.empty(pregenerated_codes_shape, code.Code)
		number_of_code_parameter_indices = np.product(pregenerated_codes_shape)
		code_generation_processes = np.empty(pregenerated_codes.shape, multiprocess.pool.ApplyResult)
		for i in range(number_of_code_parameter_indices):
			code_parameter_indices = np.unravel_index(i, pregenerated_codes_shape)
			code_parameter_indices_for_construction = code_parameter_indices
			if number_of_trials_per_parameter_set is not None:
				code_parameter_indices_for_construction = code_parameter_indices[:-1] # Last index is trial number which isn't needed for code construction.
			code_generation_processes[code_parameter_indices] = pool.apply_async(make_code_from_parameters, tuple(code_parameters[j][k] for j, k in enumerate(code_parameter_indices_for_construction)))
		for i in range(number_of_code_parameter_indices):
			code_parameter_indices = np.unravel_index(i, pregenerated_codes_shape)
			pregenerated_codes[code_parameter_indices] = code_generation_processes[code_parameter_indices].get()

		def get_single_fidelity(ec_code, noise_channel):
			lock_to_use = parameter_sweep_lock if ec_code.is_random else None
			return get_fidelity_of(ec_code, noise_channel, True, lock_to_use)

		fidelity_processes = np.empty(fidelities_shape, dtype=multiprocess.pool.AsyncResult)
		for i in range(number_of_indices):
			parameter_indices = np.unravel_index(i, noise_channels.shape)
			noise_channel = noise_channels[parameter_indices]
			if number_of_trials_per_parameter_set is None:
				ec_code = pregenerated_codes[parameter_indices[len(noise_parameters):]]
				fidelity_processes[parameter_indices] = pool.apply_async(get_single_fidelity, (ec_code, noise_channel))
			else:
				for trial_number in range(number_of_trials_per_parameter_set):
					fidelity_process_index = (*parameter_indices, trial_number)
					ec_code = pregenerated_codes[fidelity_process_index[len(noise_parameters):]]
					if number_of_trials_per_parameter_set > 1:
						assert ec_code.is_random
					fidelity_processes[fidelity_process_index] = pool.apply_async(get_single_fidelity, (ec_code, noise_channel))
		pool.close()

		for i in range(number_of_indices):
			parameter_indices = np.unravel_index(i, noise_channels.shape)
			if number_of_trials_per_parameter_set is None:
				fidelities[parameter_indices] = fidelity_processes[parameter_indices].get()
			else:
				for trial_number in range(number_of_trials_per_parameter_set):
					fidelity_process_index = (*parameter_indices, trial_number)
					fidelities[fidelity_process_index] = fidelity_processes[fidelity_process_index].get()
		pool.join()

	return fidelities
