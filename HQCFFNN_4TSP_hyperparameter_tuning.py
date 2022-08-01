import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import sympy
import itertools
from collections import Counter


###############################################################################
#                                      TSP                                    #
###############################################################################


class TSP:
    def __init__(self, number_of_cities, coords_range=(0, 10000)):
        self.number_of_cities = number_of_cities
        self.coords_range = coords_range
        self.cities_coords = self.get_cities()
        self.distance_matrix = self.calculate_distance_matrix()
        self.normalized_distance_matrix = self.normalize_distance_matrix()

    def get_cities(self):
        cities_coords = np.random.randint(
            self.coords_range[0], self.coords_range[1], size=(self.number_of_cities, 2)
        )
        return cities_coords

    def normalize_cities(self):
        max_coords = np.amax(self.cities_coords, axis=0)
        normalized_cities_coords = np.divide(self.cities_coords, max_coords)
        return normalized_cities_coords

    def calculate_distance_between_points(self, point_A, point_B):
        return np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2)

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.number_of_cities, self.number_of_cities))
        for i in range(self.number_of_cities):
            for j in range(i, self.number_of_cities):
                distance_matrix[i][j] = self.calculate_distance_between_points(
                    self.cities_coords[i], self.cities_coords[j]
                )
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    def normalize_distance_matrix(self):
        return np.divide(self.distance_matrix, np.max(self.distance_matrix))


###############################################################################
#                                     QAOA                                    #
###############################################################################


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
        self.parameters = self.generate_sympy_parameters()
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

    def generate_sympy_parameters(self):
        return sympy.symbols("parameter_:%d" % (2 * self.p))

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


###############################################################################
#                                    QFFNN                                    #
###############################################################################


class QFFNN(tf.keras.layers.Layer):
    def __init__(self, parameters):
        super(QFFNN, self).__init__()
        self.parameters = parameters
        self.params_inp = tf.keras.Input(
            shape=(len(self.parameters),), name="input_layer"
        )
        self.first_hidden = tf.keras.layers.Dense(
            len(self.parameters), name="hidden_layer"
        )
        self.expectation = tfq.layers.Expectation(name="expectation_layer")

    def call(self, inputs):
        parameterized_circuit = inputs[0]
        cost_operator = inputs[1]
        initial_parameter_values = inputs[2]

        parameter_values = self.first_hidden(initial_parameter_values)
        expectation_value = self.expectation(
            parameterized_circuit,
            operators=cost_operator,
            symbol_names=self.parameters,
            symbol_values=parameter_values,
        )

        return [parameter_values, expectation_value]


###############################################################################
#                          Hyperparameters tuning:                            #
###############################################################################


def create_hyperparams_grid():
    hyperparams = []
    for a1_value in np.arange(-1.0, 4.0, 1.0):
        a1_value = pow(10, a1_value)
        for a2_value in np.arange(-1.0, 4.0, 1.0):
            a2_value = pow(10, a2_value)
            for b_value in np.arange(-1.0, 4.0, 1.0):
                b_value = pow(10, b_value)
                hyperparams.append((a1_value, a2_value, b_value))
    return hyperparams


def choose_hyperparameters():
    return create_hyperparams_grid()


def perform_calculations(qaoa_tsp):
    parametrized_circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    operator_input = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string)
    parameters_input = tf.keras.Input(shape=(2 * qaoa_tsp.p,))

    qffnn = QFFNN(qaoa_tsp.parameters)
    output = qffnn([parametrized_circuit_input, operator_input, parameters_input])

    model = tf.keras.Model(
        inputs=[parametrized_circuit_input, operator_input, parameters_input],
        outputs=[
            output[0],  # array of optimized 2p parameters
            output[1],  # expectation value
        ],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.mean_absolute_error,
        loss_weights=[0, 1],
    )

    circuit_tensor = tfq.convert_to_tensor([qaoa_tsp.circuit])
    cost_operator_tensor = tfq.convert_to_tensor([qaoa_tsp.cost_operator])
    initial_parameters = np.zeros((1, qaoa_tsp.p * 2)).astype(np.float32)

    model.fit(
        x=[circuit_tensor, cost_operator_tensor, initial_parameters],
        y=[
            np.zeros((1, qaoa_tsp.p * 2)),
            np.zeros((1, 1)),  # the closer to 0 the better the result
        ],
        epochs=250,
        verbose=1,
    )

    parameter_values = model.predict(
        [circuit_tensor, cost_operator_tensor, initial_parameters]
    )[0]

    samples_amount = 2 ** 16
    sample_layer = tfq.layers.Sample()
    output = sample_layer(
        circuit_tensor,
        symbol_names=qaoa_tsp.parameters,
        symbol_values=parameter_values,
        repetitions=samples_amount,
    )

    results = output.numpy()[0].astype(str).tolist()
    results_to_display = ["".join(result) for result in results]
    correct_results = (
        "0001100001000010",
        "0010010010000001",
        "0100100000010010",
        "1000000100100100",
        "1000010000100001",
        "0100001000011000",
        "0001001001001000",
        "0010000110000100",
        "0100000110000010",
        "0010100000010100",
        "0001010000101000",
        "0001100000100100",
        "1000000101000010",
        "1000001001000001",
        "0100001010000001",
        "0100000100101000",
        "0010010000011000",
        "0100100000100001",
        "1000001000010100",
        "0001001010000100",
        "0001010010000010",
        "0010000101001000",
        "1000010000010010",
        "0010100001000001",
    )
    counts = Counter(results_to_display)

    correct_results_count = sum(counts[result] for result in correct_results)
    correct_results_percent = round(correct_results_count/samples_amount * 100, 2)
    return correct_results_percent


if __name__ == "__main__":
    hyperparameters = choose_hyperparameters()
    results = []
    for A_1, A_2, B in hyperparameters:
        tsp_instance = TSP(4)
        p = 10
        qaoa_tsp = QAOA_TSP(tsp_instance, p, A_1, A_2, B)
        performance = perform_calculations(qaoa_tsp)
        results.append({"hyperparameters": (A_1, A_2, B), "performance": performance})

    print("A_1\tA_2\tB\t\tperformance")
    for res in results:
        A_1, A_2, B = res["hyperparameters"]
        performance = res["performance"]
        print(f"{A_1}\t{A_2}\t{B}\t\t{performance}")

# A_1	A_2	B		performance
# 0.1	0.1	0.1		2.0
# 0.1	0.1	1.0		0.0
# 0.1	0.1	10.0		0.0
# 0.1	0.1	100.0		0.04
# 0.1	0.1	1000.0		0.02
# 0.1	1.0	0.1		10.87
# 0.1	1.0	1.0		0.83
# 0.1	1.0	10.0		0.01
# 0.1	1.0	100.0		0.0
# 0.1	1.0	1000.0		0.05

