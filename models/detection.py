from ultralytics import YOLO

def train_detection():
    try:
        model = YOLO('../yolo11n.pt')
        print("Starting YOLO11 Detection Training...")
        results = model.train(
            data='coco128.yaml',
            epochs=10,
            imgsz=640,
            device='cpu',
            patience=3,
            batch=16,
            name='detection_model'
        )
        print("Detection training done!")
        print(f"Model saved at: {results.save_dir}")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_detection():
    try:
        model = YOLO('../runs/detect/detection_model/weights/best.pt')
        print("Running detection test...")
        results = model.predict(
            source='media/bus.jpg',
            save=True
        )
        print("Prediction completed!")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    train_detection()
    test_detection()
