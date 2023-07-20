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
	TODO: document this function
	TODO: this function is SLOW; try reworking how things are looped over
	"""

	all_parameters = noise_parameters + code_parameters
	code_parameter_index_combinations = list(itertools.product(*[range(len(specific_parameter_values)) for specific_parameter_values in code_parameters]))
	code_parameter_combinations = list(itertools.product(*code_parameters))
	all_parameter_index_combinations = list(itertools.product(*[range(len(specific_parameter_values)) for specific_parameter_values in all_parameters]))
	all_parameter_combinations = list(itertools.product(*all_parameters))

	# Generate all noise channels beforehand so that there is no issue with multiprocessing. This isn't parallelized because it tends to be quick.
	noise_channels = np.empty(tuple(len(specific_parameter_values) for specific_parameter_values in all_parameters), noise.Noise)
	for parameter_indices, parameters in zip(all_parameter_index_combinations, all_parameter_combinations):
		noise_channels[parameter_indices] = make_noise_channel_from_parameters(*parameters)

	fidelities_shape = tuple(len(specific_parameter_values) for specific_parameter_values in all_parameters)
	if number_of_trials_per_parameter_set is not None:
		fidelities_shape = (*fidelities_shape, number_of_trials_per_parameter_set)
	fidelities = np.zeros(fidelities_shape)

	def initialize_pool(lock_instance):
		global parameter_sweep_lock
		parameter_sweep_lock = lock_instance
	with multiprocess.Pool(initializer=initialize_pool, initargs=(multiprocess.Lock(),)) as pool:

		# If codes are only being used once, pre-generate them because we want to ensure there are no multiprocessing issues if they're not random.
		pregenerated_codes = None
		if number_of_trials_per_parameter_set is None or number_of_trials_per_parameter_set == 1:
			pregenerated_codes = np.empty(tuple(len(code_specific_parameter_values) for code_specific_parameter_values in code_parameters), code.Code)
			code_generation_processes = np.empty(pregenerated_codes.shape, multiprocess.pool.ApplyResult)
			for parameter_indices, parameters in zip(code_parameter_index_combinations, code_parameter_combinations):
				code_generation_processes[parameter_indices] = pool.apply_async(make_code_from_parameters, parameters)
			for parameter_indices in code_parameter_index_combinations:
				pregenerated_codes[parameter_indices] = code_generation_processes[parameter_indices].get()

		def get_single_fidelity(code_parameters, code_parameter_indices, all_parameter_indices):
			ec_code = None
			if pregenerated_codes is None:
				ec_code = make_code_from_parameters(*code_parameters)
				assert ec_code.is_random
			else:
				ec_code = pregenerated_codes[code_parameter_indices]
			noise_channel = noise_channels[all_parameter_indices]
			lock_to_use = parameter_sweep_lock if ec_code.is_random else None
			return get_fidelity_of(ec_code, noise_channel, True, lock_to_use)

		fidelity_processes = np.empty(fidelities_shape, dtype=multiprocess.pool.AsyncResult)
		for parameter_indices, parameters in zip(all_parameter_index_combinations, all_parameter_combinations):
			code_parameters = parameters[len(noise_parameters):]
			code_parameter_indices = parameter_indices[len(noise_parameters):]
			if number_of_trials_per_parameter_set is None:
				fidelity_processes[parameter_indices] = pool.apply_async(get_single_fidelity, (code_parameters, code_parameter_indices, parameter_indices))
			else:
				for trial_number in range(number_of_trials_per_parameter_set):
					fidelity_process_index = (*parameter_indices, trial_number)
					fidelity_processes[fidelity_process_index] = pool.apply_async(get_single_fidelity, (code_parameters, code_parameter_indices, parameter_indices))
		pool.close()

		for parameter_indices, parameters in zip(all_parameter_index_combinations, all_parameter_combinations):
			if number_of_trials_per_parameter_set is None:
				fidelities[parameter_indices] = fidelity_processes[parameter_indices].get()
			else:
				for trial_number in range(number_of_trials_per_parameter_set):
					fidelity_process_index = (*parameter_indices, trial_number)
					fidelities[fidelity_process_index] = fidelity_processes[fidelity_process_index].get()
		pool.join()

	return fidelities
