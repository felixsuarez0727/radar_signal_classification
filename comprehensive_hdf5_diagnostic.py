import h5py
import numpy as np
import collections
import argparse
import sys
import time

def comprehensive_hdf5_analysis(file_path, analyze_signals=False):
    """
    Comprehensive analysis of large HDF5 dataset
    
    Args:
        file_path (str): Path to the HDF5 file
        analyze_signals (bool): Whether to analyze signal characteristics
    """
    try:
        print(f"Analyzing HDF5 file: {file_path}")
        start_time = time.time()
        
    with h5py.File(file_path, 'r') as hf:
        # Get all keys
        keys = list(hf.keys())
        total_signals = len(keys)
        
            print("\n" + "=" * 50)
            print(f"Total Number of Signals: {total_signals:,}")
            print("=" * 50)
            
            # Analyze keys distribution
            print("\nAnalyzing signal distribution...")
            
            # Extract all signal types (modulation and domain)
            signal_types = []
            
            # Process first 100 keys to understand structure
            for i, key in enumerate(keys[:100]):
                if i == 0:
                    print(f"\nSample Key Format: {key}")
                    first_signal = hf[key][:]
                    print(f"Sample Signal Shape: {first_signal.shape}")
                
                try:
                    key_info = eval(key)
                    mod_type = key_info[0]
                    domain = key_info[1]
                    signal_types.append((mod_type, domain))
                except:
                    print(f"Warning: Failed to parse key: {key}")
            
            # Full distribution analysis (by modulation and domain)
            print("\nProcessing all keys to calculate distribution...")
            all_signal_types = []
            for i, key in enumerate(keys):
                if i % 100000 == 0 and i > 0:
                    print(f"  Processed {i:,}/{total_signals:,} keys...")
                try:
                    key_info = eval(key)
                    mod_type = key_info[0]
                    domain = key_info[1]
                    all_signal_types.append((mod_type, domain))
                except:
                    pass
            
            # Count distribution
            distribution = collections.Counter(all_signal_types)
            
            # Print distribution table
            print("\n" + "=" * 80)
            print("SIGNAL DISTRIBUTION")
            print("=" * 80)
            print(f"{'Modulation Type':<25} | {'Domain':<30} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 80)
            
            for (mod_type, domain), count in sorted(distribution.items()):
                percentage = (count / total_signals) * 100
                print(f"{mod_type:<25} | {domain:<30} | {count:<10,} | {percentage:.2f}%")
            
            # Summary by modulation type
            mod_distribution = collections.Counter([mod for mod, _ in all_signal_types])
            print("\n" + "=" * 50)
            print("SUMMARY BY MODULATION TYPE")
            print("=" * 50)
            for mod_type, count in sorted(mod_distribution.items()):
                percentage = (count / total_signals) * 100
                print(f"{mod_type:<20} | {count:<10,} | {percentage:.2f}%")
            
            # Summary by domain
            domain_distribution = collections.Counter([domain for _, domain in all_signal_types])
            print("\n" + "=" * 50)
            print("SUMMARY BY DOMAIN")
            print("=" * 50)
            for domain, count in sorted(domain_distribution.items()):
                percentage = (count / total_signals) * 100
                print(f"{domain:<30} | {count:<10,} | {percentage:.2f}%")
            
            # Signal analysis (optional)
            if analyze_signals:
                print("\n" + "=" * 50)
                print("SIGNAL CHARACTERISTICS ANALYSIS")
                print("=" * 50)
                
                # Analyze a sample of signals
                print("\nAnalyzing signal characteristics (sampling 1000 signals)...")
                
        signal_lengths = []
                signal_channels = []
        signal_means = []
        signal_stds = []
        
                # Sample signals for analysis
                sample_keys = np.random.choice(keys, min(1000, len(keys)), replace=False)
                
                for i, key in enumerate(sample_keys):
                    if i % 100 == 0 and i > 0:
                        print(f"  Analyzed {i}/1000 signals...")
                    
            signal = hf[key][:]
            signal_lengths.append(signal.shape[0])
                    
                    if len(signal.shape) > 1:
                        signal_channels.append(signal.shape[1])
                    else:
                        signal_channels.append(1)
                    
            signal_means.append(np.mean(signal))
            signal_stds.append(np.std(signal))
        
                print("\nSignal Statistics:")
                print(f"  Average Length: {np.mean(signal_lengths):.2f} samples")
                print(f"  Length Range: {min(signal_lengths)} - {max(signal_lengths)} samples")
                
                if len(set(signal_channels)) == 1:
                    print(f"  Channels: {signal_channels[0]}")
                else:
                    print(f"  Channels: Varies ({min(signal_channels)} - {max(signal_channels)})")
                
                print(f"  Average Mean Value: {np.mean(signal_means):.6f}")
                print(f"  Average Standard Deviation: {np.mean(signal_stds):.6f}")
        
            # Dataset size estimate
            mem_estimate = 0
            for i, key in enumerate(np.random.choice(keys, min(100, len(keys)), replace=False)):
                signal = hf[key][:]
                mem_estimate += signal.nbytes
            
            avg_signal_size = mem_estimate / min(100, len(keys))
            total_estimate = avg_signal_size * total_signals
            
            print("\n" + "=" * 50)
            print("MEMORY USAGE ESTIMATE")
            print("=" * 50)
            print(f"Average Signal Size: {avg_signal_size / 1024:.2f} KB")
            print(f"Estimated Total Dataset Size: {total_estimate / (1024**3):.2f} GB")
            
            end_time = time.time()
            print("\n" + "=" * 50)
            print(f"Analysis completed in {end_time - start_time:.2f} seconds")
            print("=" * 50)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze HDF5 Dataset')
    parser.add_argument('--file', type=str, default='/Users/jaimearevalo/Downloads/RadComOta2.45GHz.hdf5',
                        help='Path to the HDF5 file')
    parser.add_argument('--analyze_signals', action='store_true',
                        help='Analyze signal characteristics (slower)')
    
    args = parser.parse_args()
    comprehensive_hdf5_analysis(args.file, args.analyze_signals)