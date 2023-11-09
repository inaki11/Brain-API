from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LSTM
from tensorflow.keras.optimizers import Adam
from cond_rnn import ConditionalRecurrent

def default_demand_forecast(parametros_arquitectura):
    input_1_shape, variables_condicionales_shape = parametros_arquitectura
    input1 = Input(shape=input_1_shape)
    input2 = Input(shape=variables_condicionales_shape)
    x = Conv1D(filters=1200, kernel_size=3, strides=3, padding="causal", activation="relu", input_shape=input_1_shape)(input1)
    x = ConditionalRecurrent(LSTM( 900, input_shape=input_1_shape, activation='tanh', return_sequences=True))([x, input2])
    x = LSTM(1100, activation='tanh')(x)
    x = Dropout(0.1)(x)
    output = Dense(24)(x)

    model = Model(inputs=[input1, input2], outputs=output)

    return model

