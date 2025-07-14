import os
import torch.optim as optim
import torch
import itertools
import pickle
from torch.utils.data import Dataset, DataLoader
from visualization import plot_training_distance_distribution, plot_results
from evaluation import evaluate_model
from processor import SignatureProcessor, create_balanced_pairs
from config import device, MODEL_CONFIG, TRAINING_CONFIG, OUTPUT_DIRECTORY
from model import ContrastiveLoss, SiameseLSTM

class SiameseDataset(Dataset):
    """PyTorch Dataset class"""
    
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        
        # Convert NumPy array to PyTorch tensor
        sig1 = torch.FloatTensor(pair[0])
        sig2 = torch.FloatTensor(pair[1])
        label = torch.FloatTensor([label])
        
        return sig1, sig2, label


def train_model(model, train_loader, val_loader):
    """Train the model"""
    lr = TRAINING_CONFIG["learning_rate"]
    margin = TRAINING_CONFIG["margin"]
    criterion = ContrastiveLoss(margin = margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.2, patience=5, 
                                                    min_lr=0.0001)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = TRAINING_CONFIG["patience"]
    num_epochs = TRAINING_CONFIG["num_epochs"]
    patience_counter = 0
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for sig1, sig2, labels in train_loader:
            sig1, sig2, labels = sig1.to(device), sig2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            distances = model(sig1, sig2)
            loss = criterion(distances, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sig1, sig2, labels in val_loader:
                sig1, sig2, labels = sig1.to(device), sig2.to(device), labels.to(device)
                
                distances = model(sig1, sig2)
                loss = criterion(distances, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses


def train_siamese_lstm(data_directory, save_model=True):
    """Main training function - trains on all pairwise combinations"""
    batch_size = TRAINING_CONFIG["batch_size"]
    try:
        # Create output folders
        subdirs = [
            "models",
            "processors",
            "thresholds",
            "model_info",
            "train_results",
            "test_results"
        ]

        base_dir = OUTPUT_DIRECTORY

        # Ana klasör ve alt klasörleri oluştur
        os.makedirs(base_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

        # Data processor
        processor = SignatureProcessor()

        # Load data
        signatures, person_labels, person_names = processor.load_signatures_from_person_folders(data_directory)

        print(f"Total signatures loaded: {len(signatures)}")
        print(f"Total number of persons: {len(person_names)}")

        # Compute normalization parameters
        processor.fit_normalizer(signatures)
        normalized_signatures = processor.normalize_signatures(signatures)

        # Determine target length
        processor.fit_target_length(normalized_signatures)

        # Interpolation
        interpolated_signatures = processor.interpolate_signatures(normalized_signatures)

        # Get all persons
        unique_persons = list(set(person_labels))

        # Pairwise combinations
        person_combinations = list(itertools.combinations(unique_persons, 2))

        print(f"A total of {len(person_combinations)} pairwise combinations will be tested:")
        for i, (person1, person2) in enumerate(person_combinations):
            print(f"  {i+1}: {person_names[person1]} - {person_names[person2]}")

        all_results = []

        for combination_idx, test_persons in enumerate(person_combinations):
            print(f"\n{'='*50}")
            print(f"Combination {combination_idx + 1}/{len(person_combinations)}")
            print(f"Selected test persons: {[person_names[p] for p in test_persons]}")
            print(f"{'='*50}")

            train_indices, test_indices = [], []

            for i, person_id in enumerate(person_labels):
                if person_id in test_persons:
                    test_indices.append(i)
                else:
                    train_indices.append(i)

            test_person_names = [person_names[p] for p in test_persons]

            X_train = [interpolated_signatures[i] for i in train_indices]
            X_test = [interpolated_signatures[i] for i in test_indices]
            y_train = [person_labels[i] for i in train_indices]
            y_test = [person_labels[i] for i in test_indices]

            print(f"Number of training signatures: {len(X_train)}")
            print(f"Number of test signatures: {len(X_test)}")
            print(f"Number of test persons: {len(set(y_test))}")

            test_person_counts = {}
            for person_id in y_test:
                test_person_counts[person_id] = test_person_counts.get(person_id, 0) + 1

            print("Number of signatures per person in test set:")
            for person_id, count in test_person_counts.items():
                print(f"  {person_names[person_id]}: {count} signatures")

            X_pairs_train, y_pairs_train = create_balanced_pairs(X_train, y_train)
            X_pairs_test, y_pairs_test = create_balanced_pairs(X_test, y_test)

            print(f"Number of training pairs: {len(X_pairs_train)}")
            print(f"Number of test pairs: {len(X_pairs_test)}")

            train_dataset = SiameseDataset(X_pairs_train, y_pairs_train)
            test_dataset = SiameseDataset(X_pairs_test, y_pairs_test)

            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

            model = SiameseLSTM(**MODEL_CONFIG)

            train_losses, val_losses = train_model(model, train_loader, test_loader)

            # Plot training distance distribution and calculate threshold
            threshold = plot_training_distance_distribution(model, train_loader, combination_idx)

            accuracy, distances, y_test_eval, predictions, precision, recall, f1 = evaluate_model(model, test_loader, threshold)
            plot_results(train_losses, val_losses, y_test_eval, distances, predictions, test_person_names, threshold)

            combination_result = {
                'combination_idx': combination_idx + 1,
                'test_persons': test_persons,
                'test_person_names': test_person_names,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'used_threshold': threshold,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            all_results.append(combination_result)

            if save_model:
                model_filename = f'{OUTPUT_DIRECTORY}/models/siamese_lstm_pytorch_combination_{combination_idx + 1}.pth'
                processor_filename = f'{OUTPUT_DIRECTORY}/processors/signature_processor_pytorch_combination_{combination_idx + 1}.pkl'
                threshold_filename = f'{OUTPUT_DIRECTORY}/thresholds/optimal_threshold_pytorch_combination_{combination_idx + 1}.pkl'
                model_info_filename = f'{OUTPUT_DIRECTORY}/model_info/model_info_combination_{combination_idx + 1}.pkl'

                torch.save(model.state_dict(), model_filename)
                processor.save_processor(processor_filename)

                model_info = {
                    'input_size': 3,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'target_length': processor.target_length,
                    'test_persons': list(test_persons),
                    'test_person_names': test_person_names,
                    'combination_idx': combination_idx + 1
                }
                with open(model_info_filename, 'wb') as f:
                    pickle.dump(model_info, f)

                with open(threshold_filename, 'wb') as f:
                    pickle.dump(threshold, f)

        print(f"\n{'='*60}")
        print("ALL COMBINATION RESULTS")
        print(f"{'='*60}")

        for result in all_results:
            print(f"Combination {result['combination_idx']}: {' - '.join(result['test_person_names'])} "
                  f"-> Accuracy: {result['accuracy']:.4f}")

        best_result = max(all_results, key=lambda x: x['accuracy'])
        worst_result = min(all_results, key=lambda x: x['accuracy'])
        avg_accuracy = sum(result['accuracy'] for result in all_results) / len(all_results)

        print("\nSUMMARY STATISTICS:")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Best Result: Combination {best_result['combination_idx']} "
              f"({' - '.join(best_result['test_person_names'])}) -> {best_result['accuracy']:.4f}")
        print(f"Worst Result: Combination {worst_result['combination_idx']} "
              f"({' - '.join(worst_result['test_person_names'])}) -> {worst_result['accuracy']:.4f}")

        if save_model:
            with open(f'{OUTPUT_DIRECTORY}/all_combinations_results.pkl', 'wb') as f:
                pickle.dump(all_results, f)
            print(f"All combination results saved to '{OUTPUT_DIRECTORY}/all_combinations_results.pkl'.")

    except Exception as e:
        print(f"Error during training: {e}")
        
    