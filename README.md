# Deployment of XGBoost model using Triton Inference Server FIL backend and Ray Serve

- [x] Training script
- [x] Triton server service launcher
- [x] Client wrapper
- [ ] Deploy model using GPU
- [ ] Ray serve deployment module
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

### Test inference

```bash
python3 infer.py
```