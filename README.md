# Deployment of XGBoost model using Triton Inference Server FIL backend and Ray Serve

- [x] Training script
- [x] Triton server service launcher
- [x] Client wrapper
- [x] Deploy model using GPU
- [x] Ray serve deployment module
- [ ] Dockerfile for client service
- [ ] Client service launcher
- [ ] Requirements.txt
- [ ] Comment, Docstring and link to references

## Train and generate model repository directory

```bash
python3 train_model.py
```

## Launch triton server

```bash
docker-compose up
```

## Launch app
```bash
./launch.sh
```

## Stop app
```bash
./stop.sh
```

### Test inference

```bash
python3 test_infer.py
```