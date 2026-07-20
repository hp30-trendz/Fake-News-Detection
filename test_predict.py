from src.models.predictor import FakeNewsPredictor

predictor = FakeNewsPredictor("random_forest")

texts = [
    "NASA discovered dinosaurs living on the moon yesterday.",
    "The Federal Reserve announced it would keep interest rates unchanged while monitoring inflation.",
]

for text in texts:
    label, confidence = predictor.predict(text)
    print("-" * 50)
    print(text)
    print("Prediction:", label)
    print("Confidence:", confidence)