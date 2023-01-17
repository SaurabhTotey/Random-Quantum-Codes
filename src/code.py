import cvxpy as cp
import numpy as np
import qutip as qt
import scipy
from typing import Optional, Tuple

def create_zero_and_one_states_from_plus_and_minus_states(plus_state: qt.Qobj, minus_state: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	zero_state = (plus_state + minus_state) / np.sqrt(2)
	one_state = (plus_state - minus_state) / np.sqrt(2)
	return (zero_state, one_state)

def create_plus_and_minus_states_from_zero_and_one_states(zero_state: qt.Qobj, one_state: qt.Qobj) -> Tuple[qt.Qobj, qt.Qobj]:
	plus_state = (zero_state + one_state) / np.sqrt(2)
	minus_state = (zero_state - one_state) / np.sqrt(2)
	return (plus_state, minus_state)

class Code:
	def __init__(self, name: str, zero_and_one_states: Tuple[qt.Qobj,  qt.Qobj], is_random: bool):
		self.name: str = name
		self.is_random: bool = is_random
		self.zero_state = zero_and_one_states[0]
		self.one_state = zero_and_one_states[1]
		assert self.zero_state.dims == self.one_state.dims
		plus_state, minus_state = create_plus_and_minus_states_from_zero_and_one_states(*zero_and_one_states)
		self.plus_state = plus_state
		self.minus_state = minus_state
		self.kraus_encoder: qt.Qobj = self.zero_state * qt.basis(2, 0).dag() + self.one_state * qt.basis(2, 1).dag()
		self.encoder = qt.sprepost(self.kraus_encoder, self.kraus_encoder.dag())
		self.decoder = qt.sprepost(self.kraus_encoder.dag(), self.kraus_encoder)
		self.physical_dimension = self.kraus_encoder.dims[0][0]
		self.code_dimension = self.kraus_encoder.dims[1][0]

	def compute_optimal_choi_recovery_matrix_through(self, channel_matrix: qt.Qobj) -> qt.Qobj:
		total_dimension = self.physical_dimension * self.code_dimension
		choi_matrix = qt.super_to_choi((1 / self.code_dimension ** 2) * (channel_matrix * self.encoder).dag())
		sdp_solution = cp.Variable((total_dimension, total_dimension), complex=True)
		objective = cp.Maximize(cp.real(cp.trace(sdp_solution @ choi_matrix)))
		cp.Problem(objective, [
			cp.partial_trace(sdp_solution, [self.physical_dimension, self.code_dimension], 1) == np.identity(self.physical_dimension),
			sdp_solution >> 0,
			sdp_solution.H == sdp_solution,
		]).solve()
		return qt.Qobj(
			scipy.sparse.csr_matrix(sdp_solution.value),
			dims=[[[self.physical_dimension], [self.code_dimension]], [[self.physical_dimension], [self.code_dimension]]],
			superrep="choi",
		)

class BinomialCode(Code):
	def __init__(self, symmetry: int, number_of_filled_levels: int, physical_dimension: int):
		assert symmetry * (number_of_filled_levels + 2) <= physical_dimension
		plus_state, minus_state = qt.Qobj(), qt.Qobj()
		for i in range(number_of_filled_levels + 2):
			i_state = np.sqrt(scipy.special.binom(number_of_filled_levels + 1, i)) * qt.basis(physical_dimension, i * symmetry)
			plus_state += i_state
			minus_state += (-1) ** i * i_state
		plus_state = plus_state.unit()
		minus_state = minus_state.unit()
		super().__init__(
			f"binomial-{symmetry},{number_of_filled_levels},{physical_dimension}",
			create_zero_and_one_states_from_plus_and_minus_states(plus_state, minus_state),
			False
		)

print(
	BinomialCode(3, 10, 36)
		.compute_optimal_choi_recovery_matrix_through(qt.kraus_to_super([qt.identity(36)]))
)
