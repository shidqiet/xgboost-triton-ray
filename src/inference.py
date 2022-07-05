# make triton_client
import os
import sys
import tritonclient.grpc as grpcclient

triton_server_url = os.getenv('TRITON_SERVER', 'localhost')
url = triton_server_url+":8001"
client = grpcclient
try:
    triton_client = client.InferenceServerClient(
        url=url)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()

import numpy as np

class IRIS_ENGINE:
    """
    Xgboost model trained on Iris Dataset Deployment Class
    """

    def __init__(self, logger=None):
        self.logger = logger

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