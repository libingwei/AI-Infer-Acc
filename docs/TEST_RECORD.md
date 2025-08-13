

## 测试结果记录

### Colab 环境测试记录
GPU T4 x 1, CUDA 12.4

#### 异步stream（非默认stream）
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


#### 默认stream
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
```

### Kaggle 环境测试记录
GPU T4 x 2, CUDA 12.6, TensorRT 10.4

#### 性能对比
```bash
cd /kaggle/working/AI-Infer-Acc
export LD_LIBRARY_PATH="/kaggle/working/tensorrt/lib:${LD_LIBRARY_PATH}"

build/bin/trt_inference models/resnet18.trt 32 
build/bin/trt_inference models/resnet18_fp16.trt 32 
build/bin/trt_inference models/resnet18_int8.trt 32 
build/bin/trt_inference models/resnet18_int8.trt 32 --use-default-stream
```
---
```bash
Running inference in FP32 mode.
Using Batch Size: 32

--- Running Performance Test ---
Number of iterations: 100

--- Performance Results ---
Total time for 100 inferences: 2536.33 ms
Average Latency: 25.3633 ms
Throughput: 1261.66 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
-0.0391306 0.114466 -1.79676 -1.2343 -0.819005 0.323965 -2.1866 -1.28767 -1.90192 -0.73148 
Running inference in FP16 mode.
Using Batch Size: 32

--- Running Performance Test ---
Number of iterations: 100

--- Performance Results ---
Total time for 100 inferences: 861.688 ms
Average Latency: 8.61688 ms
Throughput: 3713.64 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
-0.0232391 0.118225 -1.79688 -1.2334 -0.8125 0.344727 -2.18164 -1.28418 -1.90039 -0.731445 
Running inference in INT8 mode.
Using Batch Size: 32

--- Running Performance Test ---
Number of iterations: 100

--- Performance Results ---
Total time for 100 inferences: 421.845 ms
Average Latency: 4.21845 ms
Throughput: 7585.72 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
0.327693 0.266699 -1.80451 -1.42337 -0.95784 0.592519 -2.20781 -1.2707 -1.6236 -0.607254 
Running inference in INT8 mode.
Using Batch Size: 32

--- Running Performance Test ---
Number of iterations: 100
Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.

--- Performance Results ---
Total time for 100 inferences: 422.76 ms
Average Latency: 4.2276 ms
Throughput: 7569.31 FPS (frames per second)
---------------------------

Displaying first 10 output values from the last run:
0.327693 0.266699 -1.80451 -1.42337 -0.95784 0.592519 -2.20781 -1.2707 -1.6236 -0.607254 
```

#### 一致性对比，无标签数据

```bash
cd /kaggle/working/AI-Infer-Acc
export LD_LIBRARY_PATH="/kaggle/working/tensorrt/lib:${LD_LIBRARY_PATH}"

echo "trt_compare resnet18 vs resnet18_fp16"
build/bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt calibration_data 200

echo "trt_compare resnet18 vs resnet18_int8"
build/bin/trt_compare models/resnet18.trt models/resnet18_int8.trt calibration_data 200
```
---

```bash
trt_compare resnet18 vs resnet18_fp16
Evaluating on 200 images (H=224, W=224)

=== Accuracy Consistency Report ===
Samples: 200
Top-1 Agreement: 98%
Top-5 Agreement: 100%
Avg Cosine Similarity: 0.999991
Avg L2 Distance: 0.327497

trt_compare resnet18 vs resnet18_int8
Evaluating on 200 images (H=224, W=224)

=== Accuracy Consistency Report ===
Samples: 200
Top-1 Agreement: 89%
Top-5 Agreement: 100%
Avg Cosine Similarity: 0.997317
Avg L2 Distance: 5.4276

```


#### 一致性+准确性对比，ImageNet数据集有标签
```bash

cd /kaggle/working/AI-Infer-Acc
export LD_LIBRARY_PATH="/kaggle/working/tensorrt/lib:${LD_LIBRARY_PATH}"

ls /kaggle/working/AI-Infer-Acc/imagenet_val/val |& head -n 5
head -n 5 imagenet_val_labels.csv
echo "-----FP32 vs FP16-----"
build/bin/trt_compare models/resnet18.trt models/resnet18_fp16.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv --class-names imagenet_classes.tsv --inspect 10 --center-crop --imagenet-norm

echo "-----FP32 vs INT8-----"
build/bin/trt_compare models/resnet18.trt models/resnet18_int8.trt imagenet_val/val 1000 --labels imagenet_val_labels.csv --class-names imagenet_classes.tsv --inspect 10 --center-crop --imagenet-norm
```
---
```bash
ILSVRC2012_val_00000001.JPEG
ILSVRC2012_val_00000002.JPEG
ILSVRC2012_val_00000003.JPEG
ILSVRC2012_val_00000004.JPEG
ILSVRC2012_val_00000005.JPEG
ILSVRC2012_val_00000001.JPEG,65
ILSVRC2012_val_00000002.JPEG,970
ILSVRC2012_val_00000003.JPEG,230
ILSVRC2012_val_00000004.JPEG,809
ILSVRC2012_val_00000005.JPEG,516
-----FP32 vs FP16-----
Evaluating on 1000 images (H=224, W=224)
Loaded labels from imagenet_val_labels.csv, entries: 1000
Loaded class names: 1000
[Inspect] ILSVRC2012_val_00000001.JPEG | FP32: 65:sea_snake | Test: 65:sea_snake | Top5(FP32): 65:sea_snake, 58:water_snake, 63:Indian_cobra, 62:rock_python, 54:hognose_snake | Top5(Test): 65:sea_snake, 58:water_snake, 63:Indian_cobra, 62:rock_python, 54:hognose_snake
[Inspect] ILSVRC2012_val_00000002.JPEG | FP32: 795:ski | Test: 795:ski | Top5(FP32): 795:ski, 970:alp, 792:shovel, 537:dogsled, 672:mountain_tent | Top5(Test): 795:ski, 970:alp, 792:shovel, 537:dogsled, 802:snowmobile
[Inspect] ILSVRC2012_val_00000003.JPEG | FP32: 230:Shetland_sheepdog | Test: 230:Shetland_sheepdog | Top5(FP32): 230:Shetland_sheepdog, 231:collie, 157:papillon, 169:borzoi, 232:Border_collie | Top5(Test): 230:Shetland_sheepdog, 231:collie, 157:papillon, 169:borzoi, 232:Border_collie
[Inspect] ILSVRC2012_val_00000004.JPEG | FP32: 809:soup_bowl | Test: 809:soup_bowl | Top5(FP32): 809:soup_bowl, 659:mixing_bowl, 968:cup, 967:espresso, 666:mortar | Top5(Test): 809:soup_bowl, 659:mixing_bowl, 968:cup, 967:espresso, 666:mortar
[Inspect] ILSVRC2012_val_00000005.JPEG | FP32: 520:crib | Test: 520:crib | Top5(FP32): 520:crib, 831:studio_couch, 516:cradle, 721:pillow, 750:quilt | Top5(Test): 520:crib, 831:studio_couch, 516:cradle, 721:pillow, 750:quilt
[Inspect] ILSVRC2012_val_00000006.JPEG | FP32: 60:night_snake | Test: 60:night_snake | Top5(FP32): 60:night_snake, 65:sea_snake, 58:water_snake, 68:sidewinder, 67:diamondback | Top5(Test): 60:night_snake, 65:sea_snake, 58:water_snake, 68:sidewinder, 67:diamondback
[Inspect] ILSVRC2012_val_00000007.JPEG | FP32: 334:porcupine | Test: 334:porcupine | Top5(FP32): 334:porcupine, 102:echidna, 337:beaver, 342:wild_boar, 341:hog | Top5(Test): 334:porcupine, 102:echidna, 337:beaver, 342:wild_boar, 341:hog
[Inspect] ILSVRC2012_val_00000008.JPEG | FP32: 429:baseball | Test: 429:baseball | Top5(FP32): 429:baseball, 934:hotdog, 551:face_powder, 928:ice_cream, 502:clog | Top5(Test): 429:baseball, 934:hotdog, 551:face_powder, 928:ice_cream, 502:clog
[Inspect] ILSVRC2012_val_00000009.JPEG | FP32: 674:mousetrap | Test: 674:mousetrap | Top5(FP32): 674:mousetrap, 341:hog, 342:wild_boar, 998:ear, 106:wombat | Top5(Test): 674:mousetrap, 341:hog, 342:wild_boar, 998:ear, 106:wombat
[Inspect] ILSVRC2012_val_00000010.JPEG | FP32: 153:Maltese_dog | Test: 153:Maltese_dog | Top5(FP32): 153:Maltese_dog, 903:wig, 204:Lhasa, 332:Angora, 283:Persian_cat | Top5(Test): 153:Maltese_dog, 903:wig, 204:Lhasa, 332:Angora, 283:Persian_cat

=== Accuracy Consistency Report ===
Samples: 1000
Top-1 Agreement: 99.7%
Top-5 Agreement: 100%
Avg Cosine Similarity: 0.999996
Avg L2 Distance: 0.252854

=== Labeled Accuracy Report ===
Labeled Samples: 1000
FP32 Top-1: 69%, Top-5: 88.8%
Test Top-1: 68.9%, Top-5: 88.8%
Delta Top-1: -0.1 pp, Delta Top-5: 0 pp
-----FP32 vs INT8-----
Evaluating on 1000 images (H=224, W=224)
Loaded labels from imagenet_val_labels.csv, entries: 1000
Loaded class names: 1000
[Inspect] ILSVRC2012_val_00000001.JPEG | FP32: 65:sea_snake | Test: 58:water_snake | Top5(FP32): 65:sea_snake, 58:water_snake, 63:Indian_cobra, 62:rock_python, 54:hognose_snake | Top5(Test): 58:water_snake, 65:sea_snake, 50:American_alligator, 111:nematode, 49:African_crocodile
[Inspect] ILSVRC2012_val_00000002.JPEG | FP32: 795:ski | Test: 795:ski | Top5(FP32): 795:ski, 970:alp, 792:shovel, 537:dogsled, 672:mountain_tent | Top5(Test): 795:ski, 970:alp, 802:snowmobile, 537:dogsled, 672:mountain_tent
[Inspect] ILSVRC2012_val_00000003.JPEG | FP32: 230:Shetland_sheepdog | Test: 230:Shetland_sheepdog | Top5(FP32): 230:Shetland_sheepdog, 231:collie, 157:papillon, 169:borzoi, 232:Border_collie | Top5(Test): 230:Shetland_sheepdog, 231:collie, 157:papillon, 169:borzoi, 232:Border_collie
[Inspect] ILSVRC2012_val_00000004.JPEG | FP32: 809:soup_bowl | Test: 809:soup_bowl | Top5(FP32): 809:soup_bowl, 659:mixing_bowl, 968:cup, 967:espresso, 666:mortar | Top5(Test): 809:soup_bowl, 968:cup, 659:mixing_bowl, 504:coffee_mug, 925:consomme
[Inspect] ILSVRC2012_val_00000005.JPEG | FP32: 520:crib | Test: 520:crib | Top5(FP32): 520:crib, 831:studio_couch, 516:cradle, 721:pillow, 750:quilt | Top5(Test): 520:crib, 516:cradle, 431:bassinet, 831:studio_couch, 697:pajama
[Inspect] ILSVRC2012_val_00000006.JPEG | FP32: 60:night_snake | Test: 58:water_snake | Top5(FP32): 60:night_snake, 65:sea_snake, 58:water_snake, 68:sidewinder, 67:diamondback | Top5(Test): 58:water_snake, 65:sea_snake, 60:night_snake, 68:sidewinder, 52:thunder_snake
[Inspect] ILSVRC2012_val_00000007.JPEG | FP32: 334:porcupine | Test: 49:African_crocodile | Top5(FP32): 334:porcupine, 102:echidna, 337:beaver, 342:wild_boar, 341:hog | Top5(Test): 49:African_crocodile, 48:Komodo_dragon, 334:porcupine, 50:American_alligator, 377:marmoset
[Inspect] ILSVRC2012_val_00000008.JPEG | FP32: 429:baseball | Test: 429:baseball | Top5(FP32): 429:baseball, 934:hotdog, 551:face_powder, 928:ice_cream, 502:clog | Top5(Test): 429:baseball, 809:soup_bowl, 659:mixing_bowl, 631:lotion, 551:face_powder
[Inspect] ILSVRC2012_val_00000009.JPEG | FP32: 674:mousetrap | Test: 341:hog | Top5(FP32): 674:mousetrap, 341:hog, 342:wild_boar, 998:ear, 106:wombat | Top5(Test): 341:hog, 674:mousetrap, 338:guinea_pig, 448:birdhouse, 342:wild_boar
[Inspect] ILSVRC2012_val_00000010.JPEG | FP32: 153:Maltese_dog | Test: 903:wig | Top5(FP32): 153:Maltese_dog, 903:wig, 204:Lhasa, 332:Angora, 283:Persian_cat | Top5(Test): 903:wig, 283:Persian_cat, 153:Maltese_dog, 204:Lhasa, 374:langur

=== Accuracy Consistency Report ===
Samples: 1000
Top-1 Agreement: 62.1%
Top-5 Agreement: 95.1%
Avg Cosine Similarity: 0.849407
Avg L2 Distance: 45.0939

=== Labeled Accuracy Report ===
Labeled Samples: 1000
FP32 Top-1: 69%, Top-5: 88.8%
Test Top-1: 53.1%, Top-5: 77.5%
Delta Top-1: -15.9 pp, Delta Top-5: -11.3 pp

```

使用普通图片进行标定的int8模型在准确性上有明显下降，尤其是Top-1准确率从69%降至53.1%。而FP16模型与FP32模型的Top-1准确率差异较小，均在69%左右。
后面改用ImageNet的标定版本,明显改善,甚至出现比FP32有提升，Top-1从69%到69.3%, Top-5从88.8%到88.9%, 输出结果如下：

```bash
-----FP32 vs INT8-----
Evaluating on 1000 images (H=224, W=224)
Loaded labels from imagenet_val_labels.csv, entries: 1000
Loaded class names: 1000
[Inspect] ILSVRC2012_val_00000001.JPEG | FP32: 65:sea_snake | Test: 65:sea_snake | Top5(FP32): 65:sea_snake, 58:water_snake, 63:Indian_cobra, 62:rock_python, 54:hognose_snake | Top5(Test): 65:sea_snake, 58:water_snake, 62:rock_python, 54:hognose_snake, 63:Indian_cobra
[Inspect] ILSVRC2012_val_00000002.JPEG | FP32: 795:ski | Test: 795:ski | Top5(FP32): 795:ski, 970:alp, 792:shovel, 537:dogsled, 672:mountain_tent | Top5(Test): 795:ski, 970:alp, 792:shovel, 537:dogsled, 672:mountain_tent
[Inspect] ILSVRC2012_val_00000003.JPEG | FP32: 230:Shetland_sheepdog | Test: 230:Shetland_sheepdog | Top5(FP32): 230:Shetland_sheepdog, 231:collie, 157:papillon, 169:borzoi, 232:Border_collie | Top5(Test): 230:Shetland_sheepdog, 231:collie, 157:papillon, 169:borzoi, 232:Border_collie
[Inspect] ILSVRC2012_val_00000004.JPEG | FP32: 809:soup_bowl | Test: 809:soup_bowl | Top5(FP32): 809:soup_bowl, 659:mixing_bowl, 968:cup, 967:espresso, 666:mortar | Top5(Test): 809:soup_bowl, 659:mixing_bowl, 968:cup, 967:espresso, 666:mortar
[Inspect] ILSVRC2012_val_00000005.JPEG | FP32: 520:crib | Test: 520:crib | Top5(FP32): 520:crib, 831:studio_couch, 516:cradle, 721:pillow, 750:quilt | Top5(Test): 520:crib, 516:cradle, 831:studio_couch, 721:pillow, 850:teddy
[Inspect] ILSVRC2012_val_00000006.JPEG | FP32: 60:night_snake | Test: 60:night_snake | Top5(FP32): 60:night_snake, 65:sea_snake, 58:water_snake, 68:sidewinder, 67:diamondback | Top5(Test): 60:night_snake, 65:sea_snake, 58:water_snake, 68:sidewinder, 67:diamondback
[Inspect] ILSVRC2012_val_00000007.JPEG | FP32: 334:porcupine | Test: 334:porcupine | Top5(FP32): 334:porcupine, 102:echidna, 337:beaver, 342:wild_boar, 341:hog | Top5(Test): 334:porcupine, 102:echidna, 337:beaver, 342:wild_boar, 341:hog
[Inspect] ILSVRC2012_val_00000008.JPEG | FP32: 429:baseball | Test: 429:baseball | Top5(FP32): 429:baseball, 934:hotdog, 551:face_powder, 928:ice_cream, 502:clog | Top5(Test): 429:baseball, 934:hotdog, 928:ice_cream, 551:face_powder, 852:tennis_ball
[Inspect] ILSVRC2012_val_00000009.JPEG | FP32: 674:mousetrap | Test: 674:mousetrap | Top5(FP32): 674:mousetrap, 341:hog, 342:wild_boar, 998:ear, 106:wombat | Top5(Test): 674:mousetrap, 341:hog, 342:wild_boar, 998:ear, 106:wombat
[Inspect] ILSVRC2012_val_00000010.JPEG | FP32: 153:Maltese_dog | Test: 153:Maltese_dog | Top5(FP32): 153:Maltese_dog, 903:wig, 204:Lhasa, 332:Angora, 283:Persian_cat | Top5(Test): 153:Maltese_dog, 204:Lhasa, 903:wig, 283:Persian_cat, 332:Angora

=== Accuracy Consistency Report ===
Samples: 1000
Top-1 Agreement: 95.6%
Top-5 Agreement: 100%
Avg Cosine Similarity: 0.998143
Avg L2 Distance: 5.26329

=== Labeled Accuracy Report ===
Labeled Samples: 1000
FP32 Top-1: 69%, Top-5: 88.8%
Test Top-1: 69.3%, Top-5: 88.9%
Delta Top-1: 0.3 pp, Delta Top-5: 0.1 pp

```

