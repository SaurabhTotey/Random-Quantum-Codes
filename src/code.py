import os
import numpy as np
import qutip as qt
import scipy
from typing import Tuple

def create_zero_and_one_encodings_from_plus_and_minus_encodings(plus_encoding: qt.Qobj, minus_encoding: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	zero_encoding = (plus_encoding + minus_encoding) / np.sqrt(2)
	one_encoding = (plus_encoding - minus_encoding) / np.sqrt(2)
	return (zero_encoding, one_encoding)

def create_plus_and_minus_encodings_from_zero_and_one_encodings(zero_encoding: qt.Qobj, one_encoding: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	plus_encoding = (zero_encoding + one_encoding) / np.sqrt(2)
	minus_encoding = (zero_encoding - one_encoding) / np.sqrt(2)
	return (plus_encoding, minus_encoding)

class Code:
	def __init__(self, family_name: str, zero_and_one_encodings: Tuple[qt.Qobj, qt.Qobj], is_random: bool):
		self.family_name: str = family_name
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
	check_encodings(*create_plus_and_minus_encodings_from_zero_and_one_encodings(code.zero_encoding, code.one_encoding), "+", "-")

trivial_code = Code("trivial", (qt.basis(2, 0), qt.basis(2, 1)), False)

def serialize_code(code: Code) -> None:
	path = f"data/code/{code.family_name}/serialized/"
	if not os.path.exists(path):
		os.makedirs(path)
	new_path = f"{path}{len(os.listdir(path)) if code.is_random else 0}/"
	os.makedirs(new_path)
	qt.qsave(code.zero_encoding, f"{new_path}zero")
	qt.qsave(code.one_encoding, f"{new_path}one")
	with open(f"data/code/{code.family_name}/is_random.txt", "w") as file:
		file.write(f"{code.is_random}")

def deserialize_code_family(family_name: str) -> list[Code]:
	path = f"data/code/{family_name}/serialized/"
	if not os.path.exists(path):
		return []
	is_random = False
	with open(f"data/code/{family_name}/is_random.txt", "r") as file:
		is_random = file.read() == "True"
	codes = []
	for subdirectory_name in os.listdir(path):
		current_path = f"{path}{subdirectory_name}/"
		zero_encoding = qt.qload(f"{current_path}zero")
		one_encoding = qt.qload(f"{current_path}one")
		codes.append(Code(family_name, (zero_encoding, one_encoding), is_random))
	return codes

def get_binomial_code(symmetry: int, average_photon_number: int, physical_dimension: int) -> Code:
	assert symmetry * (average_photon_number + 2) <= physical_dimension
	family_name = f"binomial-{symmetry},{average_photon_number},{physical_dimension}"
	existing_versions = deserialize_code_family(family_name)
	if len(existing_versions) > 0:
		return existing_versions[-1]
	plus_encoding, minus_encoding = qt.Qobj(), qt.Qobj()
	for i in range(average_photon_number + 2):
		i_encoding = np.sqrt(scipy.special.binom(average_photon_number + 1, i)) * qt.basis(physical_dimension, i * symmetry)
		plus_encoding += i_encoding
		minus_encoding += (-1) ** i * i_encoding
	plus_encoding = plus_encoding.unit()
	minus_encoding = minus_encoding.unit()
	binomial_code = Code(
		family_name,
		create_zero_and_one_encodings_from_plus_and_minus_encodings(plus_encoding, minus_encoding),
		False
	)
	assert_code_is_good(binomial_code)
	serialize_code(binomial_code)
	return binomial_code

def make_haar_random_code(symmetry: int, average_photon_number: int, physical_dimension: int) -> Code:
	assert symmetry * (average_photon_number + 2) <= physical_dimension
	zero_projector = sum([qt.ket2dm(qt.basis(physical_dimension, i * symmetry * 2)) for i in range(average_photon_number // 2 + 1)])
	one_projector = sum([qt.ket2dm(qt.basis(physical_dimension, symmetry + i * symmetry * 2)) for i in range(average_photon_number // 2 + 1)])
	random_state = qt.rand_ket_haar(physical_dimension)
	random_code = Code(
		f"haar-random-{symmetry},{average_photon_number},{physical_dimension}",
		((zero_projector * random_state).unit(), (one_projector * random_state).unit()),
		True,
	)
	assert_code_is_good(random_code)
	return random_code
