import os
import random
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from keras.layers import Add, Dense
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


# @pytest.fixture(scope='module')
def model():
    seed = 42
    os.environ['RANDOM_SEED'] = f'{seed}'
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.get_logger().setLevel('ERROR')
    random.seed(seed)

    inp = keras.Input(shape=(10,))
    x = Dense(10)(inp)
    y = Dense(10)(inp)
    z = Dense(10)(inp)
    xy = Add()([x, y])  # 5
    xy = Add()([xy, y])  # 5
    model = keras.Model(inp, [xy, z])
    return model


# @pytest.fixture(scope='module')
def data():
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, (1000, 10))
    return X


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis'])
def test_multi_clone(model, data, backend: str):
    output_dir = str(test_root_path / f'hls4mlprj_stream_multi_clone_{backend}')
    hls_config = {'Model': {'Precision': 'fixed<32,10>', 'ReuseFactor': 1}}
    model_hls = convert_from_keras_model(
        model,
        backend=backend,
        output_dir=output_dir,
        hls_config=hls_config,
        io_type='io_stream',  # clone only happens with stream io.
    )
    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = [x.numpy() for x in model(data)]

    assert np.allclose(r_hls[0], r_keras[0], atol=5e-5, rtol=0)
    assert np.allclose(r_hls[1], r_keras[1], atol=5e-5, rtol=0)


if __name__ == '__main__':
    test_multi_clone(model(), data(), 'Vivado')
