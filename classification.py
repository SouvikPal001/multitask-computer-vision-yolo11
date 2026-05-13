from ultralytics import YOLO

def train_classification():
    try:
        model = YOLO('yolo11n-cls.pt')
        print("Starting YOLO11 Classification Training...")
        results = model.train(
            data='cifar10',
            epochs=5,
            imgsz=224,
            device='cpu',
            patience=3,
            batch=32,
            name='classification_model'
        )
        print("Classification training done!")
        print(f"Model saved at: {results.save_dir}")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_classification():
    try:
        model = YOLO('runs/classify/classification_model/weights/best.pt')
        print("Running classification test...")
        results = model.predict(
            source='media/test_img.jpg',
            save=True
        )
        print("Prediction completed!")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    train_classification()
    test_classification()
