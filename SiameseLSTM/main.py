from config import DATA_DIRECTORY
from training import train_siamese_lstm

if __name__ == "__main__":
    try:
        # Start model training loop
        train_siamese_lstm(DATA_DIRECTORY)
            
    except Exception as e:
        print(f"Main program error: {e}")