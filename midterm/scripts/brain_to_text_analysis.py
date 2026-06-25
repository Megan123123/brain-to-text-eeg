"""
Brain-to-Text Competition Data Analysis Framework
Sections 2 & 3: Data Observations and Preprocessing Steps

This script provides a comprehensive framework for analyzing neural signal data
and implementing preprocessing steps for the Brain-to-Text competition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class BrainToTextAnalyzer:
    """
    Comprehensive analyzer for Brain-to-Text competition data
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.neural_data = None
        self.text_data = None
        self.preprocessed_data = None
        self.feature_names = []
        
    def load_data(self, neural_file, text_file):
        """
        Load neural signal data and corresponding text transcripts
        
        Args:
            neural_file: Path to neural signal data (CSV/parquet)
            text_file: Path to text transcript data (CSV/parquet)
        """
        print("Loading data...")
        self.neural_data = pd.read_csv(neural_file) if neural_file.endswith('.csv') else pd.read_parquet(neural_file)
        self.text_data = pd.read_csv(text_file) if text_file.endswith('.csv') else pd.read_parquet(text_file)
        
        print(f"Neural data shape: {self.neural_data.shape}")
        print(f"Text data shape: {self.text_data.shape}")
        
    def basic_data_observations(self):
        """
        Section 2: Basic data observations and statistics
        """
        print("\n" + "="*50)
        print("SECTION 2: DATA OBSERVATIONS")
        print("="*50)
        
        if self.neural_data is None:
            print("No data loaded. Please load data first.")
            return
            
        # Basic statistics
        print("\n1. BASIC STATISTICS")
        print("-" * 30)
        print("Neural Signal Statistics:")
        print(self.neural_data.describe())
        
        # Data types and missing values
        print(f"\nData types:\n{self.neural_data.dtypes}")
        print(f"\nMissing values:\n{self.neural_data.isnull().sum().sum()}")
        
        # Signal characteristics
        print(f"\nSignal characteristics:")
        print(f"Number of channels: {self.neural_data.shape[1]}")
        print(f"Number of time points: {self.neural_data.shape[0]}")
        
        return self.neural_data.describe()
    
    def visualize_neural_signals(self, channels_to_plot=5, time_range=None):
        """
        Create visualizations of neural signals
        
        Args:
            channels_to_plot: Number of channels to visualize
            time_range: Tuple of (start, end) time points to plot
        """
        print("\n2. NEURAL SIGNAL VISUALIZATIONS")
        print("-" * 40)
        
        if self.neural_data is None:
            print("No data loaded.")
            return
            
        # Select channels to plot
        channels = self.neural_data.columns[:channels_to_plot]
        
        # Create time series plots
        fig, axes = plt.subplots(channels_to_plot, 1, figsize=(15, 3*channels_to_plot))
        if channels_to_plot == 1:
            axes = [axes]
            
        for i, channel in enumerate(channels):
            data_to_plot = self.neural_data[channel]
            if time_range:
                data_to_plot = data_to_plot[time_range[0]:time_range[1]]
            
            axes[i].plot(data_to_plot)
            axes[i].set_title(f'Channel: {channel}')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True)
            
        axes[-1].set_xlabel('Time Points')
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/neural_signals_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Signal distribution
        plt.figure(figsize=(12, 8))
        for i, channel in enumerate(channels):
            plt.subplot(2, 3, i+1)
            plt.hist(self.neural_data[channel], bins=50, alpha=0.7)
            plt.title(f'Distribution - {channel}')
            plt.xlabel('Amplitude')
            plt.ylabel('Frequency')
            
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/signal_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_signal_characteristics(self):
        """
        Analyze frequency characteristics and correlations
        """
        print("\n3. SIGNAL CHARACTERISTICS ANALYSIS")
        print("-" * 40)
        
        if self.neural_data is None:
            print("No data loaded.")
            return
            
        # Frequency analysis
        print("Frequency domain analysis...")
        sample_rate = 1000  # Assuming 1kHz sampling rate - adjust as needed
        
        # Power spectral density for first few channels
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, channel in enumerate(self.neural_data.columns[:4]):
            freqs, psd = signal.welch(self.neural_data[channel], fs=sample_rate, nperseg=1024)
            axes[i].semilogy(freqs, psd)
            axes[i].set_title(f'Power Spectral Density - {channel}')
            axes[i].set_xlabel('Frequency (Hz)')
            axes[i].set_ylabel('Power/Frequency (dB/Hz)')
            axes[i].grid(True)
            
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Channel correlations
        print("Computing channel correlations...")
        corr_matrix = self.neural_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Channel Correlation Matrix')
        plt.tight_layout()
        plt.savefig('/Users/meg/Desktop/1141datamining/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def preprocess_neural_data(self, method='standard', filter_type='bandpass', 
                              low_freq=1, high_freq=50, pca_components=None):
        """
        Section 3: Preprocessing pipeline for neural signals
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            filter_type: Filter type ('bandpass', 'lowpass', 'highpass')
            low_freq: Low frequency cutoff for bandpass filter
            high_freq: High frequency cutoff for bandpass filter
            pca_components: Number of PCA components (None for no PCA)
        """
        print("\n" + "="*50)
        print("SECTION 3: PREPROCESSING STEPS")
        print("="*50)
        
        if self.neural_data is None:
            print("No data loaded.")
            return
            
        print("\n1. SIGNAL FILTERING")
        print("-" * 20)
        
        # Apply bandpass filter to remove noise
        sample_rate = 1000  # Adjust based on your data
        nyquist = sample_rate / 2
        
        if filter_type == 'bandpass':
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            print(f"Applied bandpass filter: {low_freq}-{high_freq} Hz")
        elif filter_type == 'lowpass':
            high = high_freq / nyquist
            b, a = signal.butter(4, high, btype='low')
            print(f"Applied lowpass filter: {high_freq} Hz")
        elif filter_type == 'highpass':
            low = low_freq / nyquist
            b, a = signal.butter(4, low, btype='high')
            print(f"Applied highpass filter: {low_freq} Hz")
            
        # Apply filter to all channels
        filtered_data = self.neural_data.copy()
        for column in self.neural_data.columns:
            filtered_data[column] = signal.filtfilt(b, a, self.neural_data[column])
            
        print("\n2. NORMALIZATION")
        print("-" * 15)
        
        # Normalize the data
        if method == 'standard':
            scaler = StandardScaler()
            normalized_data = pd.DataFrame(
                scaler.fit_transform(filtered_data),
                columns=filtered_data.columns,
                index=filtered_data.index
            )
            print("Applied StandardScaler normalization")
        elif method == 'minmax':
            scaler = MinMaxScaler()
            normalized_data = pd.DataFrame(
                scaler.fit_transform(filtered_data),
                columns=filtered_data.columns,
                index=filtered_data.index
            )
            print("Applied MinMaxScaler normalization")
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            normalized_data = pd.DataFrame(
                scaler.fit_transform(filtered_data),
                columns=filtered_data.columns,
                index=filtered_data.index
            )
            print("Applied RobustScaler normalization")
            
        print("\n3. FEATURE ENGINEERING")
        print("-" * 20)
        
        # Extract time-domain features
        feature_data = self.extract_time_domain_features(normalized_data)
        
        # Extract frequency-domain features
        freq_features = self.extract_frequency_domain_features(normalized_data, sample_rate)
        feature_data = pd.concat([feature_data, freq_features], axis=1)
        
        print(f"Extracted {feature_data.shape[1]} features from {normalized_data.shape[1]} channels")
        
        print("\n4. DIMENSIONALITY REDUCTION")
        print("-" * 25)
        
        if pca_components:
            pca = PCA(n_components=pca_components)
            reduced_data = pca.fit_transform(feature_data)
            
            print(f"Applied PCA: {feature_data.shape[1]} -> {pca_components} components")
            print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance')
            plt.grid(True)
            plt.savefig('/Users/meg/Desktop/1141datamining/pca_variance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            self.preprocessed_data = pd.DataFrame(
                reduced_data,
                columns=[f'PC_{i+1}' for i in range(pca_components)]
            )
        else:
            self.preprocessed_data = feature_data
            
        print(f"Final preprocessed data shape: {self.preprocessed_data.shape}")
        
        return self.preprocessed_data
    
    def extract_time_domain_features(self, data, window_size=100):
        """
        Extract time-domain features from neural signals
        """
        features = []
        feature_names = []
        
        for column in data.columns:
            signal_data = data[column].values
            
            # Basic statistics
            features.extend([
                np.mean(signal_data),
                np.std(signal_data),
                np.var(signal_data),
                np.median(signal_data),
                np.percentile(signal_data, 25),
                np.percentile(signal_data, 75),
                np.max(signal_data) - np.min(signal_data),  # Range
                np.sum(np.abs(signal_data)),  # Total energy
            ])
            
            feature_names.extend([
                f'{column}_mean', f'{column}_std', f'{column}_var',
                f'{column}_median', f'{column}_q25', f'{column}_q75',
                f'{column}_range', f'{column}_energy'
            ])
            
            # Rolling window features
            if len(signal_data) >= window_size:
                rolling_mean = pd.Series(signal_data).rolling(window=window_size).mean()
                rolling_std = pd.Series(signal_data).rolling(window=window_size).std()
                
                features.extend([
                    np.mean(rolling_mean.dropna()),
                    np.std(rolling_mean.dropna()),
                    np.mean(rolling_std.dropna()),
                    np.std(rolling_std.dropna())
                ])
                
                feature_names.extend([
                    f'{column}_rolling_mean_mean', f'{column}_rolling_mean_std',
                    f'{column}_rolling_std_mean', f'{column}_rolling_std_std'
                ])
        
        return pd.DataFrame([features], columns=feature_names)
    
    def extract_frequency_domain_features(self, data, sample_rate, freq_bands=None):
        """
        Extract frequency-domain features from neural signals
        """
        if freq_bands is None:
            freq_bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            
        features = []
        feature_names = []
        
        for column in data.columns:
            signal_data = data[column].values
            
            # Power spectral density
            freqs, psd = signal.welch(signal_data, fs=sample_rate, nperseg=1024)
            
            # Band power features
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                total_power = np.trapz(psd, freqs)
                relative_power = band_power / total_power
                
                features.extend([band_power, relative_power])
                feature_names.extend([f'{column}_{band_name}_power', f'{column}_{band_name}_relative'])
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            features.append(spectral_centroid)
            feature_names.append(f'{column}_spectral_centroid')
            
            # Spectral rolloff (95% of power)
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]
            spectral_rolloff = freqs[rolloff_idx]
            features.append(spectral_rolloff)
            feature_names.append(f'{column}_spectral_rolloff')
        
        return pd.DataFrame([features], columns=feature_names)
    
    def feature_selection(self, target_data=None, k_best=50):
        """
        Select most informative features using statistical tests
        """
        print("\n5. FEATURE SELECTION")
        print("-" * 18)
        
        if self.preprocessed_data is None:
            print("No preprocessed data available.")
            return
            
        if target_data is not None:
            # Use target data for supervised feature selection
            selector = SelectKBest(score_func=f_classif, k=k_best)
            selected_features = selector.fit_transform(self.preprocessed_data, target_data)
            
            print(f"Selected {k_best} best features using f_classif")
            print(f"Feature scores: {selector.scores_[:10]}")  # Show top 10 scores
            
            return selected_features, selector.get_support(indices=True)
        else:
            # Use variance-based selection for unsupervised case
            feature_vars = self.preprocessed_data.var()
            selected_indices = feature_vars.nlargest(k_best).index
            
            print(f"Selected {k_best} features with highest variance")
            return self.preprocessed_data[selected_indices], selected_indices
    
    def generate_preprocessing_report(self):
        """
        Generate a comprehensive report of preprocessing steps
        """
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY REPORT")
        print("="*50)
        
        if self.preprocessed_data is None:
            print("No preprocessed data available.")
            return
            
        print(f"\nOriginal data shape: {self.neural_data.shape}")
        print(f"Preprocessed data shape: {self.preprocessed_data.shape}")
        print(f"Feature reduction: {self.neural_data.shape[1]} -> {self.preprocessed_data.shape[1]}")
        
        # Data quality metrics
        print(f"\nData Quality Metrics:")
        print(f"Missing values: {self.preprocessed_data.isnull().sum().sum()}")
        print(f"Zero variance features: {(self.preprocessed_data.var() == 0).sum()}")
        print(f"Mean feature correlation: {self.preprocessed_data.corr().abs().mean().mean():.3f}")
        
        return {
            'original_shape': self.neural_data.shape,
            'preprocessed_shape': self.preprocessed_data.shape,
            'missing_values': self.preprocessed_data.isnull().sum().sum(),
            'zero_variance': (self.preprocessed_data.var() == 0).sum()
        }

# Example usage and demonstration
def demonstrate_analysis():
    """
    Demonstrate the analysis framework with sample data
    """
    print("Brain-to-Text Competition Analysis Framework")
    print("=" * 50)
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_channels = 8
    
    # Generate synthetic neural data
    sample_data = pd.DataFrame()
    for i in range(n_channels):
        # Generate realistic neural signal with noise
        t = np.linspace(0, 1, n_samples)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
        noise = np.random.normal(0, 0.1, n_samples)
        sample_data[f'channel_{i+1}'] = signal + noise
    
    # Save sample data
    sample_data.to_csv('/Users/meg/Desktop/1141datamining/sample_neural_data.csv', index=False)
    
    # Initialize analyzer
    analyzer = BrainToTextAnalyzer()
    analyzer.neural_data = sample_data
    
    # Run analysis
    print("\nRunning data analysis...")
    analyzer.basic_data_observations()
    analyzer.visualize_neural_signals(channels_to_plot=4)
    analyzer.analyze_signal_characteristics()
    
    # Run preprocessing
    print("\nRunning preprocessing...")
    analyzer.preprocess_neural_data(method='standard', pca_components=10)
    analyzer.generate_preprocessing_report()
    
    print("\nAnalysis complete! Check the generated plots and data files.")

if __name__ == "__main__":
    demonstrate_analysis()

