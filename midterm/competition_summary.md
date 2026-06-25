
    # Brain-to-Text Competition Data Summary
    
    ## Dataset Information
    - **Source**: Card et al. (2024) - "An accurate and rapidly calibrating speech neuroprosthesis"
    - **Participant**: 45-year-old man with ALS (Amyotrophic Lateral Sclerosis)
    - **Recording**: 256 intracortical electrodes in left ventral precentral gyrus
    - **Duration**: 20 months of data collection
    - **Trials**: 11,000+ Copy Task trials
    
    ## Key Performance Metrics (from original study)
    - **Day 1**: 99.6% accuracy with 50-word vocabulary
    - **Day 2**: 90.2% accuracy with 125,000-word vocabulary  
    - **Long-term**: 97.5% accuracy sustained over 8.4 months
    - **Speed**: ~32 words per minute
    - **Usage**: 248+ cumulative hours of communication
    
    ## Data Characteristics
    - **Channels**: 256 neural recording channels
    - **Resolution**: 20ms temporal resolution
    - **Features**: Threshold crossings + spike band power (512 features total)
    - **Normalization**: Z-scored based on preceding 20 trials
    - **Split**: Train/validation/test sets available
    
    ## Competition Challenge
    - **Goal**: Decode neural signals into text/speech
    - **Input**: 256-channel neural recordings
    - **Output**: Corresponding text/speech
    - **Evaluation**: Word Error Rate (WER), Character Error Rate (CER)
    
    ## Technical Requirements
    - **Preprocessing**: Signal filtering, normalization, feature engineering
    - **Modeling**: RNN/LSTM/Transformer architectures
    - **Training**: Supervised learning with neural-text pairs
    - **Evaluation**: Cross-validation and test set performance
    