from src.analyzer import train_model

if __name__ == "__main__":
    result = train_model()
    print(f"Model saved. Hold-out accuracy: {result['accuracy']:.3f}")
    print(f"Test rows: {result['test_rows']}")
    print(result["report"])
    print("Note: this starter dataset is small and illustrative; do not present this score as real-world performance.")
