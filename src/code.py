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

trivial_code = Code("trivial", (qt.basis(2, 0), qt.basis(2, 1)), False)

class BinomialCode(Code):
	def __init__(self, symmetry: int, number_of_filled_levels: int, physical_dimension: int):
		assert symmetry * (number_of_filled_levels + 2) <= physical_dimension
		plus_encoding, minus_encoding = qt.Qobj(), qt.Qobj()
		for i in range(number_of_filled_levels + 2):
			i_encoding = np.sqrt(scipy.special.binom(number_of_filled_levels + 1, i)) * qt.basis(physical_dimension, i * symmetry)
			plus_encoding += i_encoding
			minus_encoding += (-1) ** i * i_encoding
		plus_encoding = plus_encoding.unit()
		minus_encoding = minus_encoding.unit()
		super().__init__(
			f"binomial-{symmetry},{number_of_filled_levels},{physical_dimension}",
			create_zero_and_one_encodings_from_plus_and_minus_encodings(plus_encoding, minus_encoding),
			False
		)
