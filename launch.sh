#!/bin/bash
ray start --head --object-store-memory 1073741824 --dashboard-host "0.0.0.0" --dashboard-port 8265 && \
python3 deploy.py && \
"/bin/bash"