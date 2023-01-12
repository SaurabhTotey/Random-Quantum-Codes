import numpy as np
import qutip as qt
import scipy
from typing import Tuple

def create_code_encoder_from_zero_and_one_states(zero_state: qt.Qobj, one_state: qt.Qobj) -> qt.Qobj:
	kraus_encoder = zero_state * qt.basis(2, 0).dag() + one_state * qt.basis(2, 1).dag()
	return qt.sprepost(kraus_encoder, kraus_encoder.dag())

def create_code_decoder_from_zero_and_one_states(zero_state: qt.Qobj, one_state: qt.Qobj) -> qt.Qobj:
	kraus_encoder = zero_state * qt.basis(2, 0).dag() + one_state * qt.basis(2, 1).dag()
	return qt.sprepost(kraus_encoder.dag(), kraus_encoder)

def create_zero_and_one_states_from_plus_and_minus_states(plus_state: qt.Qobj, minus_state: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	zero_state = (plus_state + minus_state) / np.sqrt(2)
	one_state = (plus_state - minus_state) / np.sqrt(2)
	return (zero_state, one_state)

def create_plus_and_minus_states_from_zero_and_one_states(zero_state: qt.Qobj, one_state: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	plus_state = (zero_state + one_state) / np.sqrt(2)
	minus_state = (zero_state - one_state) / np.sqrt(2)
	return (plus_state, minus_state)

class Code:
	def __init__(self, name: str, is_random: bool, physical_dimension: int, code_dimension: int):
		assert 0 < code_dimension and code_dimension < physical_dimension
		self.name = f"{name},{physical_dimension},{code_dimension}"
		self.is_random = is_random
		self.physical_dimension = physical_dimension
		self.code_dimension = code_dimension
		self.encoder = None
		self.decoder = None

class BinomialCode(Code):
	def __init__(self, symmetry: int, physical_dimension: int, code_dimension: int):
		assert symmetry * (code_dimension + 2) <= physical_dimension
		super().__init__(f"binomial-{symmetry}", False, physical_dimension, code_dimension)
		plus_state = qt.Qobj()
		minus_state = qt.Qobj()
		for i in range(code_dimension + 2):
			i_state = np.sqrt(scipy.special.binom(code_dimension + 1, i)) * qt.basis(physical_dimension, i * symmetry)
			plus_state += i_state
			minus_state += (-1) ** i * i_state
		plus_state = plus_state.unit()
		minus_state = minus_state.unit()
		zero_state, one_state = create_zero_and_one_states_from_plus_and_minus_states(plus_state, minus_state)
		self.encoder = create_code_encoder_from_zero_and_one_states(zero_state, one_state)
		self.decoder = create_code_decoder_from_zero_and_one_states(zero_state, one_state)
