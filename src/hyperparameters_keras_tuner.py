import tensorflow as tf
import tensorflow_quantum as tfq
import keras_tuner
import numpy as np

from TSP import TSP
from QAOA_TSP import QAOA_TSP
from QFFNN import QFFNN
from utils import generate_sympy_parameters


###############################################################################
#                              Custom hypermodel:                             #
###############################################################################

class TSPHyperModel(keras_tuner.HyperModel):

    def __init__(self, p, *args, **kwargs):
        super(TSPHyperModel, self).__init__(*args, **kwargs)
        self.p = p

    def build(self, hp):
        parametrized_circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        operator_input = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string)
        parameters_input = tf.keras.Input(shape=(2 * self.p,))

        qffnn = QFFNN(generate_sympy_parameters(self.p))
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
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )
        return model

    def fit(self, hp, model, x, y, **kwargs):
        A_1 = hp.Int("A_1", min_value=0, max_value=10, step=1)
        A_2 = hp.Int("A_2", min_value=0, max_value=10, step=1)
        # B = hp.Int("B", min_value=0, max_value=2, step=1)
        B = hp.Float("B", min_value=0, max_value=1, step=0.1)
        qaoa_tsp = QAOA_TSP(TSP(4), self.p, A_1, A_2, B)

        circuit_tensor = tfq.convert_to_tensor([qaoa_tsp.circuit])
        cost_operator_tensor = tfq.convert_to_tensor([qaoa_tsp.cost_operator])
        initial_parameters = np.zeros((1, qaoa_tsp.p * 2)).astype(np.float32)

        x = [circuit_tensor, cost_operator_tensor, initial_parameters]

        return model.fit(
            x,
            y,
            **kwargs,
        )


###############################################################################
#                          Hyperparameters tuning:                            #
###############################################################################

if __name__ == "__main__":
    p = 10

    tuner = keras_tuner.RandomSearch(
        hypermodel=TSPHyperModel(p),
        objective=keras_tuner.Objective("qffnn_mean_absolute_error", "min"),
        max_trials=3,
        executions_per_trial=1,
        overwrite=True,
        directory="hyperparameters_search_results",
        project_name="QNN",
    )

    tuner.search(
        x=[],
        y=[
            np.zeros((1, p * 2)),
            np.zeros((1, 1)),  # the closer to 0 the better the result
        ],
        epochs=250
    )

    tuner.results_summary()
