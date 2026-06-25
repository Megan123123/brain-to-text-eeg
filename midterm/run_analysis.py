"""
Quick script to run analysis on downloaded competition data
"""

from progress_report_analysis import BrainToTextProgressAnalyzer
import os
import glob

def find_data_files():
    """Find downloaded data files"""
    data_files = {
        'neural': None,
        'text': None
    }
    
    # Look for common data file patterns
    patterns = ['*.csv', '*.parquet', '*.pkl']
    
    for pattern in patterns:
        files = glob.glob(f'/Users/meg/Desktop/1141datamining/{pattern}')
        for file in files:
            filename = os.path.basename(file).lower()
            if 'neural' in filename or 'signal' in filename or 'eeg' in filename:
                data_files['neural'] = file
            elif 'text' in filename or 'transcript' in filename or 'label' in filename:
                data_files['text'] = file
    
    return data_files

def main():
    print("Brain-to-Text Competition Analysis")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = BrainToTextProgressAnalyzer()
    
    # Try to find downloaded data files
    data_files = find_data_files()
    
    if data_files['neural'] and data_files['text']:
        print(f"Found neural data: {data_files['neural']}")
        print(f"Found text data: {data_files['text']}")
        
        try:
            # Load real competition data
            analyzer.load_data(data_files['neural'], data_files['text'])
            print("Successfully loaded competition data!")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using sample data instead...")
            analyzer.load_sample_data()
    else:
        print("No competition data found. Using sample data for demonstration...")
        analyzer.load_sample_data()
    
    # Run complete analysis
    print("\nRunning complete analysis...")
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis complete!")
    print("Check the generated PNG files and CSV files for your presentation.")

if __name__ == "__main__":
    main()
