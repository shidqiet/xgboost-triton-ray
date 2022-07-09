# Deployment of XGBoost model using Triton Inference Server and Ray Serve

## Train and generate model repository directory

```bash
python3 train_model.py
```

## Build client docker image

```bash
./build.sh
```

## Launch triton server and client service

```bash
docker-compose up
```

## Test inference

```bash
python3 test_infer.py
```