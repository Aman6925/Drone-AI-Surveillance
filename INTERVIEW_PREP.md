# Interview Preparation - TAVAS Industries (AI/ML Intern)

**Golden rule:** be honest. Say *"prototype using a pretrained public model"*. Never claim defence-grade accuracy. Interviewers trust candidates who know their project's limits.

## 1. Two-minute explanation
"I built **Aerial Vision**, a prototype that analyses drone-style images and videos. A user uploads media in a Streamlit dashboard. OpenCV reads it frame by frame, a pretrained **YOLOv8** model in **PyTorch** detects people and vehicles, and the app draws boxes with confidence scores and counts objects per category. For video I enabled **ByteTrack**, which gives each object a persistent ID, so I can count *unique* objects rather than the same car 100 times. It runs on my RTX 2050 with FP16 and falls back to CPU automatically. I separated published benchmark numbers from what I actually measured, and I know the main limitation: the model is COCO-trained, so tiny objects seen from altitude are missed. My next step would be fine-tuning on an aerial dataset like VisDrone."

## 2. Five-minute technical walkthrough
1. **Problem**: too much footage, too few operators → automate detection/counting/tracking.
2. **Input**: validated uploads (extension, size, decodability). Corrupted files give a friendly error.
3. **Video handling**: `cv2.VideoCapture` reads **one frame at a time**, so RAM stays flat for long videos. Output is written with `VideoWriter` then re-encoded to H.264 so browsers can play it.
4. **Model**: YOLOv8n (3.2M params) → fits 4 GB VRAM, real-time on GPU. Settings: confidence, NMS IoU, inference size (bigger helps small objects).
5. **Device logic**: use CUDA with FP16 if available and verified, else CPU fallback; GPU OOM is caught and reported.
6. **Tracking**: `model.track(persist=True)` with ByteTrack; I keep per-track stats (first/last frame, frames seen) and draw motion trails.
7. **Analytics**: category counts, confidence histogram, objects-per-frame timeline, run history, CSV export.
8. **Evaluation honesty**: published COCO mAP is labelled as published; inference time is really measured; validation needs labelled data (coco8 only as a sanity check).
9. **Limitations / next**: fine-tune on VisDrone, SAHI tiling, ONNX/TensorRT, evaluation on a labelled aerial test set.

## 3. Concepts (beginner-friendly)
**CNN**: a network that slides small learnable filters (convolutions) over an image. Early layers detect edges, deeper layers detect parts and objects. Pooling/strides shrink the map; the final layers output predictions. Weights are learned by backpropagation + gradient descent on a loss.

**YOLO**: single-stage detector - one forward pass predicts boxes and classes everywhere → fast. Backbone (features) → neck (multi-scale fusion) → head (boxes, classes). YOLOv8 is anchor-free. NMS removes duplicate boxes.

**Classification vs detection vs tracking**
- Classification: "what is in this image?" → one label.
- Detection: "what and where?" → many boxes + labels.
- Tracking: "which detection in frame *t* is the same object as in frame *t-1*?" → IDs over time.

**IoU** = area of overlap / area of union of two boxes. Used in NMS and to decide if a prediction is correct.

**Precision / Recall / mAP**
- Precision = TP/(TP+FP) - "how many of my detections were right".
- Recall = TP/(TP+FN) - "how many real objects did I find".
- AP = area under the precision-recall curve for one class; mAP = mean over classes. mAP@50 uses IoU≥0.5, mAP@50-95 averages IoU 0.5→0.95.
- Raising the confidence threshold ↑precision, ↓recall.

**OpenCV**: library for image/video I/O and processing. Note: it uses **BGR** channel order, while Streamlit/Matplotlib expect RGB → I convert for display.

**PyTorch & CUDA**: PyTorch is the tensor/deep-learning framework (autograd, models). CUDA is NVIDIA's GPU compute platform; PyTorch moves tensors/models to the GPU (`device="cuda"`) for massively parallel maths. FP16 halves memory and speeds up inference on tensor-core GPUs. Pip's default Windows torch is CPU-only, so install the CUDA wheel.

**ByteTrack**: Kalman filter predicts positions; Hungarian matching on IoU; second pass with low-score boxes recovers occluded objects.

**ONNX / TensorRT**: ONNX = framework-neutral model format. TensorRT = NVIDIA optimiser producing a fast engine for a specific GPU.

## 4. Likely questions & model answers
1. **Why YOLO instead of Faster R-CNN?** Single-stage → much faster; good accuracy/speed trade-off for video and edge devices. Faster R-CNN is two-stage, often slightly more accurate on some tasks but slower.
2. **Why YOLOv8n?** Smallest/fastest; fits 4 GB VRAM and runs on CPU as a fallback. I offer YOLOv8s for more accuracy.
3. **Is it trained by you?** No - pretrained on COCO. My work is the pipeline, UI, tracking integration, error handling and evaluation framing. Fine-tuning is the next step.
4. **What accuracy does it have?** I don't claim one for aerial data. Published COCO mAP50-95 for YOLOv8n is 37.3 (Ultralytics). Aerial accuracy is unmeasured without labelled aerial data.
5. **Why does it miss small objects in drone images?** Few pixels per object, and COCO has mostly side-view, larger objects. Fixes: larger imgsz, SAHI tiling, fine-tune on VisDrone.
6. **What does the confidence threshold do?** Filters weak predictions; trade-off between false positives and misses.
7. **What is NMS?** Keeps the highest-scoring box and removes others overlapping it above an IoU threshold.
8. **How do you handle large videos?** Stream frame by frame, optional frame skipping/downscaling, no frame list in memory.
9. **Why did you re-encode to H.264?** OpenCV's `mp4v` output often won't play in browsers.
10. **Tracking vs detection counts?** Detection counts per-frame appearances; tracking counts unique IDs.
11. **What are ID switches?** The tracker assigns a new ID to the same object after occlusion/fast motion.
12. **How do you know the GPU is used?** The app shows the device; `torch.cuda.is_available()`; speed benchmark button measures ms/frame.
13. **What if CUDA is missing?** `resolve_device` verifies a CUDA tensor op and falls back to CPU with a message.
14. **Overfitting?** Model fits training data too well, generalises poorly; use validation split, augmentation, early stopping.
15. **What is transfer learning?** Start from pretrained weights and fine-tune on a new dataset - needs less data/time.
16. **How would you fine-tune?** `YOLO("yolov8n.pt").train(data="VisDrone.yaml", epochs=50, imgsz=640)`; check mAP on held-out val set.
17. **Why mAP and not accuracy?** Detection needs both correct class *and* location; accuracy doesn't capture localisation.
18. **How would you deploy on a drone?** Export to ONNX/TensorRT, run on a Jetson-class edge device, optimise with FP16/INT8, stream results.
19. **Ethical/safety considerations?** Surveillance has privacy and misuse risks; human-in-the-loop for decisions; bias/false positives matter.
20. **What would you improve first?** Aerial fine-tuning + proper evaluation, then SAHI and edge deployment.
21. **Difference between epoch, batch, learning rate?** Epoch = full pass over data; batch = samples per update; learning rate = update step size.
22. **What is data augmentation?** Random flips, scaling, colour shifts, mosaic - improves robustness.
23. **What bugs did you hit?** (Be truthful. Good examples: BGR vs RGB, CPU-only torch wheel on Windows, browser not playing mp4v.)

## 5. Demo script (≈3 min)
1. Dashboard → show device/model → click **Run quick demo**.
2. Image page → upload a street/aerial image → show boxes, counts, download.
3. Change confidence slider → rerun → explain trade-off.
4. Tracking page → short video (≤300 frames) → show IDs, trails, unique counts.
5. Analytics → **Benchmark** button (real FPS) → explain published vs measured.
6. Close with limitations + future work (fine-tuning, SAHI, TensorRT).

## 6. Do / Don't
✅ Say "prototype", "pretrained", "public data". ✅ Mention limitations proactively. ✅ Know every module you can point to.
❌ Don't claim mAP/accuracy you didn't measure. ❌ Don't say you trained YOLO. ❌ Don't claim it's defence-ready.
