import os
import numpy as np
import qutip as qt
import scipy
from typing import Optional, Tuple
from . import noise

def create_zero_and_one_encodings_from_plus_and_minus_encodings(plus_encoding: qt.Qobj, minus_encoding: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	zero_encoding = (plus_encoding + minus_encoding) / np.sqrt(2)
	one_encoding = (plus_encoding - minus_encoding) / np.sqrt(2)
	return (zero_encoding, one_encoding)

def create_plus_and_minus_encodings_from_zero_and_one_encodings(zero_encoding: qt.Qobj, one_encoding: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	plus_encoding = (zero_encoding + one_encoding) / np.sqrt(2)
	minus_encoding = (zero_encoding - one_encoding) / np.sqrt(2)
	return (plus_encoding, minus_encoding)

class Code:
	def __init__(self, name: str, zero_and_one_encodings: Tuple[qt.Qobj, qt.Qobj], is_random: bool):
		self.name: str = name
		self.is_random: bool = is_random
		self.zero_encoding: qt.Qobj = zero_and_one_encodings[0]
		self.one_encoding: qt.Qobj = zero_and_one_encodings[1]
		assert self.zero_encoding.dims == self.one_encoding.dims
		plus_encoding, minus_encoding = create_plus_and_minus_encodings_from_zero_and_one_encodings(*zero_and_one_encodings)
		self.plus_encoding: qt.Qobj = plus_encoding
		self.minus_encoding: qt.Qobj = minus_encoding
		self.kraus_encoder: qt.Qobj = self.zero_encoding * qt.basis(2, 0).dag() + self.one_encoding * qt.basis(2, 1).dag()
		self.encoder: qt.Qobj = qt.sprepost(self.kraus_encoder, self.kraus_encoder.dag())
		self.decoder: qt.Qobj = qt.sprepost(self.kraus_encoder.dag(), self.kraus_encoder)
		self.physical_dimension: int = self.kraus_encoder.dims[0][0]
		self.code_dimension: int = self.kraus_encoder.dims[1][0]

def assert_code_is_good(code: Code) -> None:
	def check_encodings(state_one: qt.Qobj, state_two: qt.Qobj, name_one: str, name_two: str) -> None:
		relevant_inner_products = ((state_one.dag() * state_two).tr(), state_one.norm(), state_two.norm())
		if not np.allclose(relevant_inner_products, [0, 1, 1]):
			raise Exception(f"<{name_one}|{name_two}> = {relevant_inner_products[0]} ;\t sqrt(<{name_one}|{name_one}>) = {relevant_inner_products[1]} ;\t sqrt(<{name_two}|{name_two}>) = {relevant_inner_products[2]}")
	check_encodings(code.zero_encoding, code.one_encoding, "0", "1")
	check_encodings(code.plus_encoding, code.minus_encoding, "+", "-")

trivial_code = Code("trivial", (qt.basis(2, 0), qt.basis(2, 1)), False)

def serialize_non_random_code(code: Code) -> None:
	assert not code.is_random
	path = f"data/code/{code.name}/serialized/"
	if not os.path.exists(path):
		os.makedirs(path)
	qt.qsave(code.zero_encoding, f"{path}zero")
	qt.qsave(code.one_encoding, f"{path}one")

def serialize_random_code_with_conditions(code: Code, noise: noise.Noise, use_optimal_recovery: bool) -> None:
	assert code.is_random
	path = f"data/code/random/{code.name},{noise},{use_optimal_recovery}/serialized/"
	if not os.path.exists(path):
		os.makedirs(path)
	qt.qsave(code.zero_encoding, f"{path}zero")
	qt.qsave(code.one_encoding, f"{path}one")

def deserialize_non_random_code(code_name: str) -> Optional[Code]:
	path = f"data/code/{code_name}/serialized/"
	if not os.path.exists(path):
		return None
	zero_encoding = qt.qload(f"{path}zero")
	one_encoding = qt.qload(f"{path}one")
	return Code(code_name, (zero_encoding, one_encoding), False)

def deserialize_random_code_with_conditions(code_name: str, noise: noise.Noise, use_optimal_recovery: bool) -> Optional[Code]:
	path = f"data/code/random/{code_name},{noise},{use_optimal_recovery}/serialized/"
	if not os.path.exists(path):
		return None
	zero_encoding = qt.qload(f"{path}zero")
	one_encoding = qt.qload(f"{path}one")
	return Code(code_name, (zero_encoding, one_encoding), False)

def get_binomial_code(symmetry: int, average_photon_number: int, physical_dimension: int) -> Code:
	assert symmetry * (average_photon_number + 2) <= physical_dimension
	code_name = f"binomial-{symmetry},{average_photon_number},{physical_dimension}"
	existing_code = deserialize_non_random_code(code_name)
	if existing_code is not None:
		return existing_code
	plus_encoding, minus_encoding = qt.Qobj(), qt.Qobj()
	for i in range(average_photon_number + 2):
		i_encoding = np.sqrt(scipy.special.binom(average_photon_number + 1, i)) * qt.basis(physical_dimension, i * symmetry)
		plus_encoding += i_encoding
		minus_encoding += (-1) ** i * i_encoding
	plus_encoding = plus_encoding.unit()
	minus_encoding = minus_encoding.unit()
	binomial_code = Code(
		code_name,
		create_zero_and_one_encodings_from_plus_and_minus_encodings(plus_encoding, minus_encoding),
		False
	)
	assert_code_is_good(binomial_code)
	serialize_non_random_code(binomial_code)
	return binomial_code

def get_cat_code(symmetry: int, coherent_state_value: complex, squeezing: float, physical_dimension: int) -> Code:
	code_name = f"cat-{symmetry},{coherent_state_value},{squeezing},{physical_dimension}"
	existing_code = deserialize_non_random_code(code_name)
	if existing_code is not None:
		return existing_code
	zero_encoding, one_encoding = qt.Qobj(), qt.Qobj()
	for i in range(2 * symmetry):
		angle = i * np.pi / symmetry
		displacement_operator = qt.displace(physical_dimension, coherent_state_value * np.exp(1j * angle))
		squeezing_operator = qt.squeeze(physical_dimension, squeezing * np.exp(2j * (angle - np.pi / 2)))
		blade = displacement_operator * squeezing_operator * qt.basis(physical_dimension, 0)
		zero_encoding += blade
		one_encoding += (-1) ** i * blade
	cat_code = Code(code_name, (zero_encoding.unit(), one_encoding.unit()), False)
	assert_code_is_good(cat_code)
	serialize_non_random_code(cat_code)
	return cat_code

def get_gkp_code(is_hex_lattice: bool, energy_constraint: float, physical_dimension: int) -> Code:
	code_name = f"gkp-{is_hex_lattice},{energy_constraint},{physical_dimension}"
	existing_code = deserialize_non_random_code(code_name)
	if existing_code is not None:
		return existing_code
	lowering_operator = qt.destroy(physical_dimension)
	x = (lowering_operator.dag() + lowering_operator) / np.sqrt(2)
	y = 1j * (lowering_operator.dag() - lowering_operator) / np.sqrt(2)
	x_displacement = 0
	z_displacement = 0
	if is_hex_lattice:
		x_displacement = np.sqrt(np.pi / np.sqrt(3))
		z_displacement = np.exp(2j * np.pi / 3) * x_displacement
	else:
		x_displacement = np.sqrt(np.pi / 2)
		z_displacement = 1j * x_displacement
	sx_gate = qt.displace(physical_dimension, x_displacement) ** 2
	sz_gate = qt.displace(physical_dimension, z_displacement) ** 2
	sy_gate = qt.displace(physical_dimension, x_displacement + z_displacement) ** 2
	h_gate = None
	clifford_y_gate = None
	if is_hex_lattice:
		h_gate = energy_constraint * (x ** 2 + y ** 2) - (sx_gate + sx_gate.dag() + sz_gate + sz_gate.dag() + sy_gate + sy_gate.dag())
		clifford_y_gate = qt.hadamard_transform() * qt.phasegate(np.pi / 2).dag()
	else:
		h_gate = energy_constraint * (x ** 2 + y ** 2) - (sx_gate + sx_gate.dag() + sz_gate + sz_gate.dag())
		clifford_y_gate = qt.hadamard_transform()
	_, h_eigenvectors = h_gate.eigenstates(eigvals=2)
	_, c_eigenvectors = clifford_y_gate.eigenstates(eigvals=2)
	ground_state = h_eigenvectors[0]
	first_excited_state = -h_eigenvectors[1]
	u = (c_eigenvectors[1] * qt.basis(2, 0).dag() + c_eigenvectors[0] * qt.basis(2, 1).dag()).full()
	zero_encoding = u[0, 0] * ground_state + u[0, 1] * first_excited_state
	one_encoding = u[1, 0] * ground_state + u[1, 1] * first_excited_state
	gkp_code = Code(code_name, (zero_encoding, one_encoding), False)
	assert_code_is_good(gkp_code)
	serialize_non_random_code(gkp_code)
	return gkp_code

def make_projected_haar_random_code(symmetry: int, average_photon_number: int, physical_dimension: int) -> Code:
	assert symmetry * (average_photon_number + 2) <= physical_dimension
	zero_projector = sum([qt.ket2dm(qt.basis(physical_dimension, i * symmetry * 2)) for i in range(average_photon_number // 2 + 1)])
	one_projector = sum([qt.ket2dm(qt.basis(physical_dimension, symmetry + i * symmetry * 2)) for i in range(average_photon_number // 2 + 1)])
	random_state = qt.rand_ket_haar(physical_dimension)
	random_code = Code(
		f"projected-haar-random-{symmetry},{average_photon_number},{physical_dimension}",
		((zero_projector * random_state).unit(), (one_projector * random_state).unit()),
		True,
	)
	assert_code_is_good(random_code)
	return random_code

def make_expanded_haar_random_code(symmetry: int, average_photon_number: int, physical_dimension: int) -> Code:
	assert symmetry * (average_photon_number + 2) <= physical_dimension
	state_to_expand = qt.rand_ket_haar(average_photon_number // 2 + 1)
	zero_state = sum([qt.basis(physical_dimension, i * symmetry * 2) * state_to_expand[i][0][0] for i in range(average_photon_number // 2 + 1)])
	one_state = sum([qt.basis(physical_dimension, symmetry + i * symmetry * 2) * state_to_expand[i][0][0] for i in range(average_photon_number // 2 + 1)])
	random_code = Code(
		f"expanded-haar-random-{symmetry},{average_photon_number},{physical_dimension}",
		(zero_state, one_state),
		True,
	)
	assert_code_is_good(random_code)
	return random_code

def make_two_expanded_haar_random_code(symmetry: int, average_photon_number: int, physical_dimension: int) -> Code:
	assert symmetry * (average_photon_number + 2) <= physical_dimension
	zero_state_to_expand = qt.rand_ket_haar(average_photon_number // 2 + 1)
	one_state_to_expand = qt.rand_ket_haar(average_photon_number // 2 + 1)
	zero_state = sum([qt.basis(physical_dimension, i * symmetry * 2) * zero_state_to_expand[i][0][0] for i in range(average_photon_number // 2 + 1)])
	one_state = sum([qt.basis(physical_dimension, symmetry + i * symmetry * 2) * one_state_to_expand[i][0][0] for i in range(average_photon_number // 2 + 1)])
	random_code = Code(
		f"two-expanded-haar-random-{symmetry},{average_photon_number},{physical_dimension}",
		(zero_state, one_state),
		True,
	)
	assert_code_is_good(random_code)
	return random_code
