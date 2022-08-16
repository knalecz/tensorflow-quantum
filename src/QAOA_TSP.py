import cirq
import tensorflow_quantum as tfq
import numpy as np
import itertools
from utils import generate_sympy_parameters


class QAOA_TSP:
    def __init__(self, tsp_instance, p=1, A_1=4, A_2=4, B=1):
        self.tsp_instance = tsp_instance
        self.qubits = cirq.GridQubit.rect(1, tsp_instance.number_of_cities ** 2)
        self.p = p
        self.weights = {
            "cost_weight": B,
            "constraint_each_visited": A_1,
            "constraint_each_visited_once": A_2,
        }
        self.parameters = generate_sympy_parameters(p)
        self.cost_operator = self.create_cost_operator()
        self.circuit = self.create_qaoa_circuit()

    def calc_bit(self, i, t):
        return i + t * self.tsp_instance.number_of_cities

    def x(self, i, t):
        x = self.calc_bit(i, t)
        qubit = self.qubits[x]
        return cirq.PauliString(0.5, cirq.I(qubit)) - cirq.PauliString(
            0.5, cirq.Z(qubit)
        )

    def create_hadamard_circuit_layer(self):
        hadamard_circuit_layer = cirq.Circuit()
        for qubit in self.qubits:
            hadamard_circuit_layer += cirq.H(qubit)
        return hadamard_circuit_layer

    def create_cost_operator(self):
        A_1 = self.weights["constraint_each_visited"]
        A_2 = self.weights["constraint_each_visited_once"]
        B = self.weights["cost_weight"]

        cost_of_constraint_each_visited = 0
        for i in range(self.tsp_instance.number_of_cities):
            curr = 1
            for t in range(self.tsp_instance.number_of_cities):
                curr -= self.x(i, t)
            cost_of_constraint_each_visited += np.power(curr, 2)

        cost_of_constraint_each_visited_once = 0
        for t in range(self.tsp_instance.number_of_cities):
            curr = 1
            for i in range(self.tsp_instance.number_of_cities):
                curr -= self.x(i, t)
            cost_of_constraint_each_visited_once += np.power(curr, 2)

        cost_of_visiting_cities = 0
        for i, j in itertools.permutations(
            range(0, self.tsp_instance.number_of_cities), 2
        ):
            curr = 0
            for t in range(self.tsp_instance.number_of_cities):
                inc_t = t + 1
                if inc_t == self.tsp_instance.number_of_cities:
                    inc_t = 0
                curr += self.x(i, t) * self.x(j, inc_t)
            cost_of_visiting_cities += (
                self.tsp_instance.normalized_distance_matrix[i][j] * curr
            )

        cost_operator = (
            A_1 * cost_of_constraint_each_visited
            + A_2 * cost_of_constraint_each_visited_once
            + B * cost_of_visiting_cities
        )

        return cost_operator

    def create_mixing_operator(self):
        mixing_operator = 0
        for qubit in self.qubits:
            mixing_operator += cirq.X(qubit)
        return mixing_operator

    def create_qaoa_circuit(self):
        hadamard_circuit_layer = self.create_hadamard_circuit_layer()
        cost_operator = self.create_cost_operator()
        mixing_operator = self.create_mixing_operator()
        parameterized_circuit_layers = tfq.util.exponential(
            operators=[cost_operator, mixing_operator] * self.p,
            coefficients=self.parameters,
        )
        qaoa_circuit = hadamard_circuit_layer + parameterized_circuit_layers
        return qaoa_circuit
