# Deployment of XGBoost model using Triton Inference Server and Ray Serve
* Deploy XGBoost model using Triton Inference Server FIL backend
* Use Ray Serve to:
    * Handle inference request (HTTP)
    * Preprocess request
    * Pass on the processed request using tritonclient
    * Postprocess and returning inference result

To use this repository, you can follow these steps:

1. Train and generate model repository directory

```bash
python3 train_model.py
```

In this repository I am not too focused on the model development process, so i just try to train xgboost model on iris dataset.

2. Build client docker image

```bash
./build.sh
```

3. Launch triton server and client service

```bash
docker-compose up
```

4. Test inference

```bash
python3 test_infer.py
```