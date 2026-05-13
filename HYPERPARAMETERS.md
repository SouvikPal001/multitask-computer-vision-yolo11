# YOLO11 Hyperparameters Explained

## What I Used:

### epochs = 3/5/10
- **What it is:** Number of times model sees entire dataset
- **Why 3/5/10:** Fast training for demo (would be 100+ for production)
- **Trade-off:** More epochs = better accuracy but slower training

### imgsz = 640
- **What it is:** Input image resolution
- **Why 640:** Standard YOLO size, good balance
- **Trade-off:** Larger (1024) = more detail but slower. Smaller (224) = faster but less accurate

### batch = 16
- **What it is:** Images processed at once
- **Why 16:** Good balance between speed and memory
- **Trade-off:** Larger batch = faster training but needs more GPU memory

### device = 'cpu'
- **What it is:** Which GPU/CPU to use
- **Why 'cpu':** Uses CPU for training (slower than training on GPU, use when GPU is not available(0 for First GPU and 1 for Second GPU))

### patience = 3
- **What it is:** Early stopping - stop if no improvement for 3 epochs
- **Why 3:** Quick demo (would be 15-20 for full training)
- **Trade-off:** Higher = train longer. Lower = stop early if stuck

## Key Learnings:

1. Different tasks (detection, segmentation, pose) need different models
2. Hyperparameters control speed vs accuracy
3. Real deployment would need:
   - Larger epochs (50-100)
   - Custom data (not COCO)
   - More hyperparameter tuning
   - Hours/days of training


.