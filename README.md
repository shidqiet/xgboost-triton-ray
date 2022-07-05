# Deployment of XGBoost model using Triton Inference Server FIL backend and Ray Serve

- [x] Training script
- [x] Triton server service launcher
- [x] Client wrapper
- [x] Deploy model using GPU
- [x] Ray serve deployment module
- [x] Dockerfile for client service
- [ ] Client service launcher
- [x] Requirements.txt
- [ ] Comment, Docstring and link to references

## Train and generate model repository directory

```bash
python3 train_model.py
```

## Launch triton server

```bash
docker-compose up
```

## Build app docker image

```bash
./build.sh
```

## Launch app service

```bash
./deploy.sh
```

## Test inference

```bash
python3 test_infer.py
```