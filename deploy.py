import argparse
import logging

import ray
from ray import serve
from fastapi import FastAPI

from src.inference import IRIS_ENGINE
from src.template import RequestData, InferData

import numpy as np

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")

@serve.deployment(route_prefix="/api")
@serve.ingress(app)
class IRIS_APP:
    """
    XGBoost model trained on Iris Dataset Deployment Class
    """
    def __init__(self):
        self.logger = logging.getLogger("ray")

        # init iris engine
        self.iris_engine = IRIS_ENGINE(logger=self.logger)
        
    def log(self, msg):
        """Logging"""
        self.logger.info(msg)

    @app.post("/infer")
    async def req_infer(self, request: RequestData):
        """
        Receive inference data requests, validate the data,and then inference it

        PARAMETERS
        ----------
        request: RequestData
            A model with four fields: sepal_length, sepal_width,
            petal_length, and petal_width
        
        Returns
        -------
        response: InferData
            A model with three fields: confidence score for iris_setosa, 
            iris_versicolour, iris_virginica
        """
        _requests = [
            request.sepal_length, 
            request.sepal_width, 
            request.petal_length, 
            request.petal_width
        ]
        requests = np.asarray(_requests).astype(np.float32)
        requests = np.expand_dims(requests, axis=0)
        _response = self.iris_engine.infer([requests])
        response = InferData(
            iris_setosa = _response[0],
            iris_versicolour = _response[1],
            iris_virginica = _response[2]
        )
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="0.0.0.0")
    parser.add_argument(
        "--port",
        default=6060)
    args = parser.parse_args()

    ray.init(address="auto", namespace="iris")
    http_config = {
            'host': args.host,
            'port': args.port,
        }

    client = serve.start(detached=True, http_options=http_config)
    IRIS_APP.deploy()
