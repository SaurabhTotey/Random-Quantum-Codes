import cvxpy as cp
import numpy as np
import qutip as qt
import scipy
from typing import Optional, Tuple

def create_kraus_encoder_from_zero_and_one_states(zero_state: qt.Qobj, one_state: qt.Qobj) -> qt.Qobj:
	return zero_state * qt.basis(2, 0).dag() + one_state * qt.basis(2, 1).dag()

def create_zero_and_one_states_from_plus_and_minus_states(plus_state: qt.Qobj, minus_state: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	zero_state = (plus_state + minus_state) / np.sqrt(2)
	one_state = (plus_state - minus_state) / np.sqrt(2)
	return (zero_state, one_state)

def create_plus_and_minus_states_from_zero_and_one_states(zero_state: qt.Qobj, one_state: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	plus_state = (zero_state + one_state) / np.sqrt(2)
	minus_state = (zero_state - one_state) / np.sqrt(2)
	return (plus_state, minus_state)

class Code:
	def __init__(self, name: str, is_random: bool, physical_dimension: int):
		assert 0 < physical_dimension
		self.name: str = f"{name},{physical_dimension}"
		self.is_random: bool = is_random
		self.physical_dimension: int = physical_dimension
		self.kraus_encoder: Optional[qt.Qobj] = None

	def get_encoder(self) -> Optional[qt.Qobj]:
		return qt.sprepost(self.kraus_encoder, self.kraus_encoder.dag()) if self.kraus_encoder is not None else None

	def get_decoder(self) -> Optional[qt.Qobj]:
		return qt.sprepost(self.kraus_encoder.dag(), self.kraus_encoder) if self.kraus_encoder is not None else None

	def compute_optimal_choi_recovery_matrix_through(self, channel_matrix: qt.Qobj) -> Optional[qt.Qobj]:
		if self.kraus_encoder is None:
			return None
		code_dimension = self.kraus_encoder.dims[1][0]
		total_dimension = self.physical_dimension * code_dimension
		choi_matrix = qt.super_to_choi((1 / code_dimension ** 2) * (channel_matrix * self.get_encoder()).dag())
		sdp_solution = cp.Variable((total_dimension, total_dimension), complex=True)
		objective = cp.Maximize(cp.real(cp.trace(sdp_solution @ choi_matrix)))
		cp.Problem(objective, [
			cp.partial_trace(sdp_solution, [self.physical_dimension, code_dimension], 1) == np.identity(self.physical_dimension),
			sdp_solution >> 0,
			sdp_solution.H == sdp_solution,
		]).solve()
		return qt.Qobj(
			scipy.sparse.csr_matrix(sdp_solution.value),
			dims=[[[self.physical_dimension], [code_dimension]], [[self.physical_dimension], [code_dimension]]],
			superrep="choi",
		)

class BinomialCode(Code):
	def __init__(self, symmetry: int, physical_dimension: int, code_dimension: int):
		assert symmetry * (code_dimension + 2) <= physical_dimension
		super().__init__(f"binomial-{symmetry}/{code_dimension}", False, physical_dimension)
		plus_state = qt.Qobj()
		minus_state = qt.Qobj()
		for i in range(code_dimension + 2):
			i_state = np.sqrt(scipy.special.binom(code_dimension + 1, i)) * qt.basis(physical_dimension, i * symmetry)
			plus_state += i_state
			minus_state += (-1) ** i * i_state
		plus_state = plus_state.unit()
		minus_state = minus_state.unit()
		zero_state, one_state = create_zero_and_one_states_from_plus_and_minus_states(plus_state, minus_state)
		self.kraus_encoder = create_kraus_encoder_from_zero_and_one_states(zero_state, one_state)

print(
	BinomialCode(3, 36, 10)
		.compute_optimal_choi_recovery_matrix_through(qt.kraus_to_super([qt.identity(36)]))
)
