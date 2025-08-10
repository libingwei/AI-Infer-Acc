

## 测试结果记录

### 异步stream（非默认stream）
```bash
/content/drive/MyDrive/AI-Infer-Acc# ./bin/trt_inference models/resnet18_int8.trt 32
Running inference in INT8 mode.
Using Batch Size: 32
Using non-default CUDA stream: 0x5b9a4793e470

--- Running Performance Test ---
Number of iterations: 100

--- Performance Results ---
Total time for 100 inferences: 477.567 ms
Average Latency: 4.77567 ms
Throughput: 6700.63 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
0.419745 0.354462 -1.69596 -1.24819 -0.809193 0.704928 -2.14123 -1.26328 -1.66599 -0.823498 

/content/drive/MyDrive/AI-Infer-Acc# ./bin/trt_inference models/resnet18_fp16.trt 32
Running inference in FP16 mode.
Using Batch Size: 32
Using non-default CUDA stream: 0x5a89462493a0

--- Running Performance Test ---
Number of iterations: 100

--- Performance Results ---
Total time for 100 inferences: 852.68 ms
Average Latency: 8.5268 ms
Throughput: 3752.87 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
-0.025116 0.115723 -1.79785 -1.2334 -0.813965 0.343018 -2.17969 -1.28516 -1.89551 -0.734863 

/content/drive/MyDrive/AI-Infer-Acc# ./bin/trt_inference models/resnet18.trt 32
Running inference in FP32 mode.
Using Batch Size: 32
Using non-default CUDA stream: 0x5ad9ba3abae0

--- Running Performance Test ---
Number of iterations: 100

--- Performance Results ---
Total time for 100 inferences: 2578.14 ms
Average Latency: 25.7814 ms
Throughput: 1241.2 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
-0.039132 0.114464 -1.79676 -1.2343 -0.819005 0.323964 -2.1866 -1.28767 -1.90192 -0.731483 
```


### 默认stream
```bash
/content/drive/MyDrive/AI-Infer-Acc# ./bin/trt_inference models/resnet18_int8.trt 32
Running inference in INT8 mode.
Using Batch Size: 32

--- Running Performance Test ---
Number of iterations: 100
Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.

--- Performance Results ---
Total time for 100 inferences: 450.868 ms
Average Latency: 4.50868 ms
Throughput: 7097.42 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
0.419745 0.354462 -1.69596 -1.24819 -0.809193 0.704928 -2.14123 -1.26328 -1.66599 -0.823498 

/content/drive/MyDrive/AI-Infer-Acc# ./bin/trt_inference models/resnet18_int8.trt 1 
Running inference in INT8 mode.
Using Batch Size: 1

--- Running Performance Test ---
Number of iterations: 100
Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.

--- Performance Results ---
Total time for 100 inferences: 80.6246 ms
Average Latency: 0.806246 ms
Throughput: 1240.32 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
0.419745 0.354462 -1.69596 -1.24819 -0.809193 0.704928 -2.14123 -1.26328 -1.66599 -0.823498 