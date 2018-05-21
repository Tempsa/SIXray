#!/usr/bin/env bash

th main.lua -optim sgd -LR 0.01 -verbose true -netType googlenet-inception4e -epochStep 25 -nClasses 5 -batchSize 32 -nGPU 1 -gpuDevice '{1}' -imageSize 224 -data /root/Dataset/X-ray-dataset/ -dataset  X-ray-cls -loss MultiLabel -train multilabel  -k 15 -nEpochs 60
