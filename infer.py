import sys
import logging

import numpy as np

# make triton_client
import tritonclient.grpc as grpcclient
url = "localhost:8001"
client = grpcclient
try:
    triton_client = client.InferenceServerClient(
        url=url)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()

def add_log_handler(logger: logging.Logger, log_level: int):
    """
    Handlers will be added to the logger:
    - StreamHandler
    """

    log_formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s",
        "%y%m%d %H:%M:%S")
    logger_stream_handler = logging.StreamHandler()
    logger_stream_handler.setLevel(log_level)
    logger_stream_handler.setFormatter(log_formatter)
    logger.addHandler(logger_stream_handler)

class IRIS_ENGINE:
    """
    Xgboost model trained on Iris Dataset Deployment Class
    """

    def __init__(self):
        self.logger = logging.Logger("IRIS_ENGINE")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        add_log_handler(self.logger, logging.INFO)

        self.model_name = 'iris_xgboost'
        self.model_version = '1'
        self.init_triton()

    def init_triton(self):
        """
        Triton input and output initiation
        """
        # input config and data shape
        input_names = ['input__0']
        input_shapes = [(1, 4)]
        # output config
        output_names = ['output__0']
        # Set the input and output of TRIS model
        self.inputs = [ 
            client.InferInput(
                input_name, input_shape, 'FP32') for input_name, input_shape in zip(input_names, input_shapes)
        ]
        self.outputs = [ 
            client.InferRequestedOutput(output_name) for output_name in output_names
        ]

    def infer(
            self, 
            data: np.ndarray) -> np.ndarray:
        """
        Inference
        """
        for _input, _data in zip(self.inputs, data):
            _input.set_data_from_numpy(_data)
        response = triton_client.infer(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        inputs=self.inputs,
                        outputs=self.outputs)
        prediction = response.as_numpy('output__0')[0]
        return prediction

if __name__ == "__main__":
    iris_engine = IRIS_ENGINE()
    print(iris_engine.infer(np.zeros(shape=(1, 1, 4), dtype=np.float32)))