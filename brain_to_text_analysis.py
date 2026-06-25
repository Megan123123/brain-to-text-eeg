"""
Brain-to-Text Competition Analysis Framework
For Progress Report Sections 2 & 3: Data Observations and Preprocessing

Based on Card et al. (2024) dataset: "An accurate and rapidly calibrating speech neuroprosthesis"
Dataset: https://doi.org/10.5061/dryad.dncjsxm85
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class BrainToTextAnalyzer:
    """
    Comprehensive analyzer for Brain-to-Text competition data
    Based on Card et al. (2024) research
    """
    
    def __init__(self):
        self.neural_data = None
        self.text_data = None
        self.preprocessed_data = None
        self.sampling_rate = 1000  # 20ms resolution = 50Hz, but we'll use 1000Hz for analysis
        
    def load_sample_data(self, n_samples=2000, n_channels=256):
        """
        Generate realistic sample neural data based on Card et al. (2024) specifications
        - 256 intracortical electrodes
        - 20ms temporal resolution
        - Threshold crossings + spike band power features
        """
        print("Generating sample neural data based on Card et al. (2024) specifications...")
        print("- 256 intracortical electrodes")
        print("- 20ms temporal resolution")
        print("- Threshold crossings + spike band power features")
        
        # Create time vector (20ms resolution)
        t = np.linspace(0, n_samples/50, n_samples)  # 50Hz sampling rate
        
        # Generate realistic neural signals with different characteristics
        neural_signals = {}
        
        for i in range(n_channels):
            # Different frequency bands for different electrode locations
            if i < 64:  # Delta/Theta channels (slower rhythms)
                freq1, freq2 = 2, 6
                amplitude = 0.8
            elif i < 128:  # Alpha channels
                freq1, freq2 = 8, 12
                amplitude = 1.2
            elif i < 192:  # Beta channels
                freq1, freq2 = 13, 25
                amplitude = 1.0
            else:  # Gamma channels
                freq1, freq2 = 30, 50
                amplitude = 0.6
            
            # Generate base signal with speech-related modulation
            signal_base = (amplitude * np.sin(2 * np.pi * freq1 * t) + 
                           0.7 * amplitude * np.sin(2 * np.pi * freq2 * t) +
                           0.3 * amplitude * np.sin(2 * np.pi * (freq1 + freq2) * t))
            
            # Add speech-related modulation (simulating attempted speech)
            speech_modulation = 1 + 0.5 * np.sin(2 * np.pi * 0.1 * t) * np.random.random()
            
            # Add realistic noise and artifacts
            noise = np.random.normal(0, 0.1, n_samples)
            artifacts = np.random.normal(0, 0.05, n_samples) * np.random.poisson(0.1, n_samples)
            
            # Combine all components
            neural_signals[f'electrode_{i+1:03d}'] = speech_modulation * signal_base + noise + artifacts
        
        self.neural_data = pd.DataFrame(neural_signals)
        
        # Generate corresponding text data (simplified)
        words = ['hello', 'world', 'brain', 'signal', 'text', 'decode', 'neural', 'activity', 
                'speech', 'communication', 'interface', 'computer', 'technology', 'research']
        self.text_data = pd.DataFrame({
            'word': np.random.choice(words, n_samples//100),
            'start_time': np.arange(0, n_samples, 100),
            'end_time': np.arange(100, n_samples+100, 100),
            'confidence': np.random.uniform(0.7, 1.0, n_samples//100)
        })
        
        print(f"Generated neural data: {self.neural_data.shape}")
        print(f"Generated text data: {self.text_data.shape}")
        
        return self.neural_data, self.text_data
    
    def section2_data_observations(self):
        """
        Section 2: Comprehensive data observations and analysis
        """
        print("\n" + "="*60)
        print("SECTION 2: DATA OBSERVATIONS")
        print("="*60)
        
        if self.neural_data is None:
            print("No data loaded. Generating sample data...")
            self.load_sample_data()
        
        # 1. Basic Data Statistics
        print("\n1. BASIC DATA STATISTICS")
        print("-" * 30)
        
        stats_summary = {
            'Number of electrodes': self.neural_data.shape[1],
            'Number of time points': self.neural_data.shape[0],
            'Temporal resolution': '20ms',
            'Duration (seconds)': self.neural_data.shape[0] / 50,  # 50Hz sampling
            'Missing values': self.neural_data.isnull().sum().sum(),
            'Data type': str(self.neural_data.dtypes.iloc[0])
        }
        
        for key, value in stats_summary.items():
            print(f"{key}: {value}")
        
        # Detailed statistics for first 10 electrodes
        print(f"\nElectrode Statistics (first 10):")
        print(self.neural_data.iloc[:, :10].describe().round(3))
        
        # 2. Signal Quality Assessment
        print("\n2. SIGNAL QUALITY ASSESSMENT")
        print("-" * 35)
        
        # Calculate signal-to-noise ratio
        signal_power = np.var(self.neural_data, axis=0)
        noise_estimate = np.var(np.diff(self.neural_data, axis=0), axis=0)
        snr = 10 * np.log10(signal_power / noise_estimate)
        
        print(f"Signal-to-Noise Ratio (dB) - First 10 electrodes:")
        for i, electrode in enumerate(self.neural_data.columns[:10]):
            print(f"  {electrode}: {snr[i]:.2f} dB")
        
        # 3. Visualizations
        self._create_data_visualizations()
        
        # 4. Frequency Analysis
        self._analyze_frequency_characteristics()
        
        # 5. Electrode Correlations
        self._analyze_electrode_correlations()
        
        return self.analysis_results if hasattr(self, 'analysis_results') else {}
    
    def _create_data_visualizations(self):
        """
        Create comprehensive visualizations for data observations
        """
        print("\n3. DATA VISUALIZATIONS")
        print("-" * 25)
        
        # Time series plots for first 8 electrodes
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, electrode in enumerate(self.neural_data.columns[:8]):
            # Plot first 500 samples for clarity
            data_to_plot = self.neural_data[electrode].iloc[:500]
            axes[i].plot(data_to_plot, linewidth=0.8)
            axes[i].set_title(f'Electrode {electrode} (First 10s)', fontsize=10)
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Time Points (20ms resolution)')
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/neural_signals_timeseries.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Signal distributions
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, electrode in enumerate(self.neural_data.columns[:8]):
            axes[i].hist(self.neural_data[electrode], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution - {electrode}')
            axes[i].set_xlabel('Amplitude (μV)')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/signal_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots for electrode comparison
        plt.figure(figsize=(15, 8))
        # Show first 20 electrodes for clarity
        self.neural_data.iloc[:, :20].boxplot(figsize=(15, 8))
        plt.title('Electrode Amplitude Distributions (First 20)')
        plt.ylabel('Amplitude (μV)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/electrode_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_frequency_characteristics(self):
        """
        Analyze frequency domain characteristics
        """
        print("\n4. FREQUENCY DOMAIN ANALYSIS")
        print("-" * 30)
        
        # Power spectral density analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        freq_bands = {
            'Delta (1-4 Hz)': (1, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-50 Hz)': (30, 50)
        }
        
        band_powers = {band: [] for band in freq_bands.keys()}
        
        for i, electrode in enumerate(self.neural_data.columns[:4]):
            # Calculate PSD
            freqs, psd = signal.welch(self.neural_data[electrode], 
                                    fs=50, nperseg=256)  # 50Hz sampling rate
            
            # Plot PSD
            axes[i].semilogy(freqs, psd, linewidth=1)
            axes[i].set_title(f'Power Spectral Density - {electrode}')
            axes[i].set_xlabel('Frequency (Hz)')
            axes[i].set_ylabel('Power/Frequency (dB/Hz)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 25)  # Focus on relevant frequency range
            
            # Calculate band powers
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers[band_name].append(band_power)
        
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/frequency_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Band power analysis
        band_power_df = pd.DataFrame(band_powers, 
                                   index=self.neural_data.columns[:4])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(band_power_df, annot=True, cmap='viridis', fmt='.2e')
        plt.title('Frequency Band Powers Across Electrodes')
        plt.ylabel('Electrodes')
        plt.xlabel('Frequency Bands')
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/frequency_band_powers.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Frequency band analysis completed.")
        print("Key findings:")
        for band in freq_bands.keys():
            mean_power = np.mean(band_powers[band])
            print(f"  {band}: Mean power = {mean_power:.2e}")
    
    def _analyze_electrode_correlations(self):
        """
        Analyze correlations between electrodes
        """
        print("\n5. ELECTRODE CORRELATION ANALYSIS")
        print("-" * 35)
        
        # Calculate correlation matrix for first 20 electrodes
        corr_matrix = self.neural_data.iloc[:, :20].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Electrode Correlation Matrix (First 20)')
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/electrode_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated electrode pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                          corr_matrix.columns[j], corr_val))
        
        print(f"Found {len(high_corr_pairs)} highly correlated electrode pairs (|r| > 0.7):")
        for pair in high_corr_pairs[:5]:  # Show top 5
            print(f"  {pair[0]} - {pair[1]}: r = {pair[2]:.3f}")
    
    def section3_preprocessing_steps(self):
        """
        Section 3: Comprehensive preprocessing pipeline
        """
        print("\n" + "="*60)
        print("SECTION 3: PREPROCESSING STEPS")
        print("="*60)
        
        if self.neural_data is None:
            print("No data loaded. Generating sample data...")
            self.load_sample_data()
        
        # 1. Data Cleaning
        print("\n1. DATA CLEANING")
        print("-" * 15)
        
        # Check for missing values
        missing_before = self.neural_data.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Remove outliers using IQR method
        cleaned_data = self.neural_data.copy()
        for column in cleaned_data.columns:
            Q1 = cleaned_data[column].quantile(0.25)
            Q3 = cleaned_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data[column] = cleaned_data[column].clip(lower_bound, upper_bound)
        
        outliers_removed = (self.neural_data != cleaned_data).sum().sum()
        print(f"Outliers clipped: {outliers_removed}")
        
        # 2. Signal Filtering
        print("\n2. SIGNAL FILTERING")
        print("-" * 18)
        
        filtered_data = self._apply_signal_filters(cleaned_data)
        
        # 3. Normalization
        print("\n3. NORMALIZATION")
        print("-" * 15)
        
        normalized_data = self._normalize_signals(filtered_data)
        
        # 4. Feature Engineering
        print("\n4. FEATURE ENGINEERING")
        print("-" * 20)
        
        engineered_features = self._engineer_features(normalized_data)
        
        # 5. Feature Selection
        print("\n5. FEATURE SELECTION")
        print("-" * 18)
        
        selected_features = self._select_features(engineered_features)
        
        # 6. Dimensionality Reduction
        print("\n6. DIMENSIONALITY REDUCTION")
        print("-" * 25)
        
        reduced_data = self._reduce_dimensions(selected_features)
        
        self.preprocessed_data = reduced_data
        
        # Generate preprocessing report
        self._generate_preprocessing_report()
        
        return self.preprocessed_data
    
    def _apply_signal_filters(self, data):
        """
        Apply various signal filters
        """
        print("Applying signal filters...")
        
        # Bandpass filter (1-50 Hz) - appropriate for neural signals
        nyquist = 25  # 50Hz sampling rate / 2
        low = 1 / nyquist
        high = 50 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered_data = data.copy()
        for column in data.columns:
            filtered_data[column] = signal.filtfilt(b, a, data[column])
        
        print("  - Applied bandpass filter (1-50 Hz)")
        
        # Notch filter for power line noise (50/60 Hz)
        notch_freq = 50  # Adjust based on your region
        b_notch, a_notch = signal.iirnotch(notch_freq, Q=30, fs=50)
        
        for column in data.columns:
            filtered_data[column] = signal.filtfilt(b_notch, a_notch, filtered_data[column])
        
        print(f"  - Applied notch filter ({notch_freq} Hz)")
        
        # Visualize filtering effects
        self._visualize_filtering_effects(data, filtered_data)
        
        return filtered_data
    
    def _visualize_filtering_effects(self, original, filtered):
        """
        Visualize the effects of filtering
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, electrode in enumerate(original.columns[:4]):
            # Time domain
            axes[i].plot(original[electrode].iloc[:500], alpha=0.7, label='Original')
            axes[i].plot(filtered[electrode].iloc[:500], label='Filtered')
            axes[i].set_title(f'Filtering Effect - {electrode}')
            axes[i].set_xlabel('Time Points')
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/filtering_effects.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _normalize_signals(self, data):
        """
        Apply normalization techniques
        """
        print("Applying normalization...")
        
        # Z-score normalization
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        print("  - Applied Z-score normalization")
        
        # Visualize normalization effects
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Before normalization
        data.iloc[:, :4].boxplot(ax=axes[0])
        axes[0].set_title('Before Normalization')
        axes[0].set_ylabel('Amplitude (μV)')
        
        # After normalization
        normalized_data.iloc[:, :4].boxplot(ax=axes[1])
        axes[1].set_title('After Normalization')
        axes[1].set_ylabel('Normalized Amplitude')
        
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/normalization_effects.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return normalized_data
    
    def _engineer_features(self, data):
        """
        Extract comprehensive features from neural signals
        """
        print("Engineering features...")
        
        features_list = []
        feature_names = []
        
        for column in data.columns:
            signal_data = data[column].values
            
            # Time domain features
            time_features = [
                np.mean(signal_data),           # Mean
                np.std(signal_data),            # Standard deviation
                np.var(signal_data),            # Variance
                np.median(signal_data),         # Median
                np.percentile(signal_data, 25), # Q1
                np.percentile(signal_data, 75), # Q3
                np.max(signal_data) - np.min(signal_data),  # Range
                np.sum(np.abs(signal_data)),    # Total energy
                np.sqrt(np.mean(signal_data**2)),  # RMS
                np.mean(np.abs(signal_data)),   # Mean absolute value
            ]
            
            time_names = [
                f'{column}_mean', f'{column}_std', f'{column}_var',
                f'{column}_median', f'{column}_q25', f'{column}_q75',
                f'{column}_range', f'{column}_energy', f'{column}_rms', f'{column}_mav'
            ]
            
            features_list.extend(time_features)
            feature_names.extend(time_names)
            
            # Frequency domain features
            freqs, psd = signal.welch(signal_data, fs=50, nperseg=256)
            
            freq_bands = {
                'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
                'beta': (13, 30), 'gamma': (30, 50)
            }
            
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                total_power = np.trapz(psd, freqs)
                relative_power = band_power / total_power
                
                features_list.extend([band_power, relative_power])
                feature_names.extend([f'{column}_{band_name}_power', 
                                    f'{column}_{band_name}_relative'])
            
            # Spectral features
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            spectral_rolloff_idx = np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0][0]
            spectral_rolloff = freqs[spectral_rolloff_idx]
            
            features_list.extend([spectral_centroid, spectral_rolloff])
            feature_names.extend([f'{column}_spectral_centroid', 
                                f'{column}_spectral_rolloff'])
        
        engineered_features = pd.DataFrame([features_list], columns=feature_names)
        print(f"  - Extracted {len(feature_names)} features")
        
        return engineered_features
    
    def _select_features(self, features):
        """
        Select most informative features
        """
        print("Selecting features...")
        
        # Remove features with zero variance
        variance_threshold = 1e-10
        feature_vars = features.var()
        selected_mask = feature_vars > variance_threshold
        selected_features = features.loc[:, selected_mask]
        
        print(f"  - Removed {sum(~selected_mask)} zero-variance features")
        print(f"  - Selected {selected_features.shape[1]} features")
        
        return selected_features
    
    def _reduce_dimensions(self, features):
        """
        Apply dimensionality reduction
        """
        print("Applying dimensionality reduction...")
        
        # PCA
        n_components = min(50, features.shape[1])
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(features)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"  - Applied PCA: {features.shape[1]} -> {n_components} components")
        print(f"  - Explained variance: {explained_variance:.3f}")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/pca_explained_variance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return pd.DataFrame(reduced_data, 
                          columns=[f'PC_{i+1}' for i in range(n_components)])
    
    def _generate_preprocessing_report(self):
        """
        Generate comprehensive preprocessing report
        """
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY REPORT")
        print("="*50)
        
        if self.preprocessed_data is None:
            print("No preprocessed data available.")
            return
        
        print(f"\nData Pipeline Summary:")
        print(f"  Original shape: {self.neural_data.shape}")
        print(f"  Preprocessed shape: {self.preprocessed_data.shape}")
        print(f"  Feature reduction: {self.neural_data.shape[1]} -> {self.preprocessed_data.shape[1]}")
        print(f"  Compression ratio: {self.preprocessed_data.shape[1]/self.neural_data.shape[1]:.2f}")
        
        print(f"\nData Quality Metrics:")
        print(f"  Missing values: {self.preprocessed_data.isnull().sum().sum()}")
        print(f"  Zero variance features: {(self.preprocessed_data.var() == 0).sum()}")
        print(f"  Mean feature correlation: {self.preprocessed_data.corr().abs().mean().mean():.3f}")
        
        # Save preprocessed data
        self.preprocessed_data.to_csv('/Users/meg/Desktop/1141datamining/preprocessed_data.csv', index=False)
        print(f"\nPreprocessed data saved to: preprocessed_data.csv")
        
        return {
            'original_shape': self.neural_data.shape,
            'preprocessed_shape': self.preprocessed_data.shape,
            'compression_ratio': self.preprocessed_data.shape[1]/self.neural_data.shape[1],
            'missing_values': self.preprocessed_data.isnull().sum().sum(),
            'zero_variance': (self.preprocessed_data.var() == 0).sum()
        }
    
    def run_complete_analysis(self):
        """
        Run the complete analysis for both sections
        """
        print("Starting complete Brain-to-Text analysis...")
        print("Based on Card et al. (2024) dataset specifications")
        print("="*60)
        
        # Section 2: Data Observations
        self.section2_data_observations()
        
        # Section 3: Preprocessing Steps
        self.section3_preprocessing_steps()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files for your progress report:")
        print("  - neural_signals_timeseries.png (time series plots)")
        print("  - signal_distributions.png (amplitude distributions)")
        print("  - electrode_distributions.png (box plots)")
        print("  - frequency_analysis.png (power spectral density)")
        print("  - frequency_band_powers.png (band power analysis)")
        print("  - electrode_correlations.png (correlation matrix)")
        print("  - filtering_effects.png (preprocessing effects)")
        print("  - normalization_effects.png (normalization effects)")
        print("  - pca_explained_variance.png (dimensionality reduction)")
        print("  - preprocessed_data.csv (final processed data)")
        
        return True

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BrainToTextAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis framework ready for your progress report!")
    print("Use the generated visualizations and data for your presentation.")