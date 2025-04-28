import h5py
import numpy as np
import collections

def comprehensive_hdf5_analysis(file_path):
    """
    Comprehensive analysis of large HDF5 dataset
    """
    with h5py.File(file_path, 'r') as hf:
        # Get all keys
        keys = list(hf.keys())
        total_signals = len(keys)
        
        # Analyze modulation types and domains
        modulation_types = collections.Counter()
        domains = collections.Counter()
        unique_signal_types = set()
        
        # For memory and signal characteristics
        total_memory = 0
        signal_lengths = []
        signal_means = []
        signal_stds = []
        
        # Analyze first 1000 signals to get a representative sample
        for key in keys[:1000]:
            # Parse the key
            signal_type = eval(key)
            mod_type, domain = signal_type[:2]
            
            # Count modulation types and domains
            modulation_types[mod_type] += 1
            domains[domain] += 1
            unique_signal_types.add(signal_type)
            
            # Signal characteristics
            signal = hf[key][:]
            total_memory += signal.nbytes
            signal_lengths.append(signal.shape[0])
            signal_means.append(np.mean(signal))
            signal_stds.append(np.std(signal))
        
        # Print comprehensive analysis
        print("HDF5 Dataset Comprehensive Analysis")
        print("=" * 40)
        print(f"Total Number of Signals: {total_signals}")
        
        # Modulation Types
        print("\nModulation Types:")
        for mod_type, count in modulation_types.items():
            percentage = (count / 1000) * 100
            print(f"  - {mod_type}: {count} ({percentage:.2f}%)")
        
        # Domains
        print("\nSignal Domains:")
        for domain, count in domains.items():
            percentage = (count / 1000) * 100
            print(f"  - {domain}: {count} ({percentage:.2f}%)")
        
        # Signal Characteristics
        print("\nSignal Characteristics (from first 1000 signals):")
        print(f"  Average Signal Length: {np.mean(signal_lengths):.2f}")
        print(f"  Signal Length Range: {min(signal_lengths)} - {max(signal_lengths)}")
        print(f"  Average Signal Mean: {np.mean(signal_means):.6f}")
        print(f"  Average Signal Std Dev: {np.mean(signal_stds):.6f}")
        
        # Memory Usage
        print(f"\nTotal Dataset Memory Usage: {total_memory / (1024 * 1024):.2f} MB")
        
        # Unique Signal Types
        print("\nUnique Signal Types (first 20):")
        for sig_type in list(unique_signal_types)[:20]:
            print(f"  - {sig_type}")
        
        # Optional: More detailed signal type distribution
        print("\nDetailed Unique Signal Types Distribution:")
        signal_type_counts = collections.Counter(tuple(eval(key)[:2]) for key in keys)
        for (mod_type, domain), count in signal_type_counts.items():
            percentage = (count / total_signals) * 100
            print(f"  - {mod_type} in {domain}: {count} ({percentage:.2f}%)")

# Path to your HDF5 file
file_path = '/Users/jaimearevalo/Downloads/RadComOta2.45GHz.hdf5'
comprehensive_hdf5_analysis(file_path)