#!/bin/bash
cd "D:\jiangli\123\tiaoshi 为了让奖励不为0"

echo "[RUN] Starting training..."
python train.py \
    task=ShadowHandHora \
    train=ShadowHandHora \
    headless=True \
    2>&1 | tee /tmp/hora_run.log

echo "[RUN] Log saved to /tmp/hora_run.log"
