# Deployment of XGBoost model using Triton Inference Server FIL backend and Ray Serve

- [x] Training script
- [x] Triton server service launcher
- [x] Client wrapper
- [x] Deploy model using GPU
- [x] Ray serve deployment module
- [x] Dockerfile for client service
- [x] Client service launcher
- [x] Requirements.txt
- [ ] Comment, Docstring and link to references
- [ ] Make a decent explanation in Readme

## Train and generate model repository directory

```bash
python3 train_model.py
```

## Build app docker image

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