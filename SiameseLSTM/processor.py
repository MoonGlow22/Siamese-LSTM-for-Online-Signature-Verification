import os
import pandas as pd
import numpy as np
import glob
import pickle
import random
from scipy import interpolate
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from config import TRAINING_CONFIG

class SignatureProcessor:
    """Loads, normalizes and interpolates the signatures"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.target_length = None
    
    def load_signature_from_file(self, file_path):
        try:
            if os.path.getsize(file_path) == 0:
                print(f"Warning: {file_path} is an empty file")
                return None
                
            data = pd.read_csv(file_path, header=None, sep=',', skipinitialspace=True)
            
            if data.empty:
                print(f"Warning: {file_path} is empty")
                return None
                
            if data.shape[1] >= 3:
                # Extract coordinates and timestamps separately
                xy_coords = data.iloc[:, :2].values
                timestamps = data.iloc[:, 2].values
                
                # Calculate time differences
                dt = np.diff(timestamps, prepend=timestamps[0])
                
                # Combine coordinates with time differences
                signature = np.column_stack((xy_coords, dt))
                
                # Check for NaN and inf values
                if np.any(np.isnan(signature)) or np.any(np.isinf(signature)):
                    print(f"Warning: Invalid values found in {file_path}")
                    return None
                
                # Check minimum number of points
                if len(signature) < 5:
                    print(f"Warning: {file_path} contains too few points ({len(signature)} points)")
                    return None
                    
                return signature
            else:
                print(f"Warning: {file_path} does not have enough columns")
                return None
                
        except Exception as e:
            print(f"Error while reading file {file_path}: {e}")
            return None
    
    def load_signatures_from_person_folders(self, data_directory):
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        all_signatures = []
        person_labels = []
        
        person_folders = [f for f in os.listdir(data_directory) 
                         if os.path.isdir(os.path.join(data_directory, f))]
        
        if not person_folders:
            raise ValueError(f"No person folders found in: {data_directory}")
        
        valid_persons = 0
        
        for person_idx, person_name in enumerate(person_folders):
            person_path = os.path.join(data_directory, person_name)
            txt_files = glob.glob(os.path.join(person_path, "*.txt"))
            
            person_signatures = []
            
            for file_path in txt_files:
                sig = self.load_signature_from_file(file_path)
                if sig is not None:
                    person_signatures.append(sig)
            
            # Check minimum number of signatures per person
            if len(person_signatures) >= 2:  # At least 2 signatures required
                all_signatures.extend(person_signatures)
                person_labels.extend([person_idx] * len(person_signatures))
                valid_persons += 1
            else:
                print(f"Warning: Not enough signatures for {person_name} ({len(person_signatures)} signatures)")
        
        if valid_persons < 2:
            raise ValueError("At least 2 people must have at least 2 signatures each!")
        
        return all_signatures, person_labels, person_folders
    
    def fit_normalizer(self, signatures):
        """Calculate normalization parameters"""
        # Merge all signatures
        all_points = []
        for sig in signatures:
            all_points.extend(sig)
        
        all_points = np.array(all_points)
        
        # Fit the scaler
        self.scaler.fit(all_points)
        self.is_fitted = True
    
    def normalize_signatures(self, signatures):
        if not self.is_fitted:
            raise ValueError("Call fit_normalizer() first!")
        
        normalized = []
        for sig in signatures:
            if len(sig) > 0:
                sig_norm = self.scaler.transform(sig)
                normalized.append(sig_norm)
        
        return normalized
    
    def fit_target_length(self, signatures):
        """Determine target length (2x the length of the longest signature)"""
        lengths = [len(sig) for sig in signatures]
        max_length = np.max(lengths)
        self.target_length = max_length * 2
        
        # Show statistics
        print("Signature length statistics:")
        print(f"  Min: {np.min(lengths)}")
        print(f"  Max: {max_length}")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Interpolation target: {self.target_length}")
    
    def interpolate_signature(self, signature, target_length):
        """Interpolate a single signature to the target length"""
        current_length = len(signature)
        
        if current_length == target_length:
            return signature
        
        # Original indices
        original_indices = np.linspace(0, current_length - 1, current_length)
        
        # Target indices
        target_indices = np.linspace(0, current_length - 1, target_length)
        
        # Interpolation for each dimension
        interpolated_signature = np.zeros((target_length, signature.shape[1]))
        
        for dim in range(signature.shape[1]):
            # Linear interpolation
            f = interpolate.interp1d(original_indices, signature[:, dim], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            interpolated_signature[:, dim] = f(target_indices)
        
        return interpolated_signature
    
    def interpolate_signatures(self, signatures):
        """Interpolate all signatures to the same target length"""
        if self.target_length is None:
            raise ValueError("Call fit_target_length() first!")
        
        interpolated = []
        
        for sig in signatures:
            interpolated_sig = self.interpolate_signature(sig, self.target_length)
            interpolated.append(interpolated_sig)
        
        return np.array(interpolated)
    
    def save_processor(self, filepath):
        """Save the processor"""
        processor_data = {
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'target_length': self.target_length
        }
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
    
    def load_processor(self, filepath):
        """Load the processor"""
        try:
            with open(filepath, 'rb') as f:
                processor_data = pickle.load(f)
            
            self.scaler = processor_data['scaler']
            self.is_fitted = processor_data['is_fitted']
            self.target_length = processor_data['target_length']
            print(f"Processor loaded: {filepath}")
        except Exception as e:
            raise Exception(f"Error while loading processor: {e}")


def create_balanced_pairs(signatures, person_labels):
    """Create balanced positive and negative pairs"""
    max_pairs_per_person = TRAINING_CONFIG["max_pairs_per_person"]

    # Group signatures by person
    person_signatures = defaultdict(list)
    for i, person_id in enumerate(person_labels):
        person_signatures[person_id].append(signatures[i])
    
    pairs = []
    labels = []
    
    # 1. POSITIVE PAIRS
    positive_count = 0
    for person_id, person_sigs in person_signatures.items():
        if len(person_sigs) > 1:
            person_positive_count = 0
            
            for i in range(len(person_sigs)):
                for j in range(i+1, len(person_sigs)):
                    if person_positive_count >= max_pairs_per_person:
                        break
                    
                    pairs.append([person_sigs[i], person_sigs[j]])
                    labels.append(0)
                    person_positive_count += 1
                    positive_count += 1
                
                if person_positive_count >= max_pairs_per_person:
                    break
    
    # 2. NEGATIVE PAIRS (same number as positives)
    person_ids = list(person_signatures.keys())
    negative_count = 0
    target_negative = positive_count
    
    attempts = 0
    max_attempts = target_negative * 3  # Prevent infinite loop
    
    while negative_count < target_negative and attempts < max_attempts:
        person1_id = random.choice(person_ids)
        person2_id = random.choice(person_ids)
        
        if person1_id != person2_id:
            sig1 = random.choice(person_signatures[person1_id])
            sig2 = random.choice(person_signatures[person2_id])
            
            pairs.append([sig1, sig2])
            labels.append(1)
            negative_count += 1
        
        attempts += 1
    
    # Shuffle
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    
    return list(pairs), list(labels)

