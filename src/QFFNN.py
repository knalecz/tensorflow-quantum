import tensorflow as tf
import tensorflow_quantum as tfq


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
