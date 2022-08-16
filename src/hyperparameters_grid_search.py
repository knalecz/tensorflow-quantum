import csv
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from collections import Counter

from TSP import TSP
from QAOA_TSP import QAOA_TSP
from QFFNN import QFFNN


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

    history = model.fit(
        x=[circuit_tensor, cost_operator_tensor, initial_parameters],
        y=[
            np.zeros((1, qaoa_tsp.p * 2)),
            np.zeros((1, 1)),  # the closer to 0 the better the result
        ],
        epochs=25,
        verbose=1,
    )
    print(history.history)

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
    correct_results_percent = round(correct_results_count / samples_amount * 100, 2)
    return correct_results_percent


if __name__ == "__main__":
    hyperparameters = choose_hyperparameters()
    results = []

    with open("grid_search_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["A_1", "A_2", "B", "performance"])

        for A_1, A_2, B in hyperparameters:
            tsp_instance = TSP(4)
            p = 10
            qaoa_tsp = QAOA_TSP(tsp_instance, p, A_1, A_2, B)
            performance = perform_calculations(qaoa_tsp)
            results.append(
                {"hyperparameters": (A_1, A_2, B), "performance": performance}
            )
            writer.writerow([A_1, A_2, B, performance])
            print(f"{A_1}\t{A_2}\t{B}\t\t{performance}")
            break
