import sys

import coremltools as coremltools

from car_model import get_cars_model

print("Exporting Core ML model")
model = get_cars_model()

model.load_weights("checkpoints/weights_1527545864.E019-0.095.hdf5")

coreml_model = coremltools.converters.keras.convert(model, input_names='input_1', image_input_names='input_1',
                                                    output_names='car_class')

coreml_model.save("models/CarModel.mlmodel")

