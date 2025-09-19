# Audio Classification Training Notebooks

This directory contains three separate training notebooks for different state-of-the-art models on the ESC-50 audio classification dataset.

## 🚀 Quick Start

### Prerequisites
- Kaggle account with GPU access
- ESC-50 dataset uploaded to your Kaggle workspace
- GitHub repository with your codebase

### Setup
1. Upload the ESC-50 dataset to your Kaggle workspace
2. Choose one of the three notebooks below
3. **Important**: Update the GitHub URL in the first cell of each notebook to point to your repository

## 📚 Available Models

### 1. Liquid S4 (`liquid_s4_training.py`)
- **Model**: Liquid S4 (State Space Model)
- **Memory**: ~8GB GPU
- **Training Time**: 2-3 hours
- **Best For**: Efficient sequence modeling with good performance

**Setup**: Upload `liquid_s4_training.ipynb` to Kaggle or copy cells from `liquid_s4_training.py`

### 2. Mamba (`mamba_training.py`)
- **Model**: Mamba (Selective State Space Model)
- **Memory**: ~12GB GPU
- **Training Time**: 3-4 hours
- **Best For**: High-performance sequence modeling

**Setup**: Upload `mamba_training.ipynb` to Kaggle or copy cells from `mamba_training.py`

**⚠️ Important**: Mamba requires specific package versions. The notebook includes the exact installation commands from the screenshot.

### 3. VJEPA2 (`vjepa2_training.py`)
- **Model**: Video Joint Embedding Predictive Architecture 2
- **Memory**: ~10GB GPU
- **Training Time**: 2-3 hours
- **Best For**: Vision transformer adapted for audio

**Setup**: Upload `vjepa2_training.ipynb` to Kaggle or copy cells from `vjepa2_training.py`

## 📁 File Structure

```
notebooks/
├── README.md                    # This file
├── shared_utils.py              # Shared utilities for all notebooks
├── liquid_s4_training.ipynb     # Liquid S4 training notebook
├── mamba_training.ipynb         # Mamba training notebook
├── vjepa2_training.ipynb        # VJEPA2 training notebook
├── liquid_s4_training.py        # Liquid S4 training cells (backup)
├── mamba_training.py            # Mamba training cells (backup)
└── vjepa2_training.py           # VJEPA2 training cells (backup)
```

## 🔧 How to Use

### Option 1: Upload Notebooks (Recommended)
1. **Open Kaggle**: Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. **Enable GPU**: In notebook settings, enable GPU (T4 or better)
3. **Upload ESC-50 Dataset**: Upload the ESC-50 dataset to your Kaggle workspace
4. **Upload Notebook**: Upload the `.ipynb` file for your chosen model
5. **Update GitHub URL**: Change `https://github.com/your-username/audio-classifier.git` to your actual repository URL
6. **Run Cells**: Execute each cell in order
7. **Monitor Training**: Watch the progress bars and metrics
8. **Save Results**: Models are automatically saved to `/kaggle/working/`

### Option 2: Copy Cells
1. **Open Kaggle**: Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. **Enable GPU**: In notebook settings, enable GPU (T4 or better)
3. **Upload ESC-50 Dataset**: Upload the ESC-50 dataset to your Kaggle workspace
4. **Clone Repository**: Add `!git clone https://github.com/your-username/audio-classifier.git` as the first cell
5. **Copy Code**: Copy the cells from your chosen training `.py` file
6. **Run Cells**: Execute each cell in order
7. **Monitor Training**: Watch the progress bars and metrics
8. **Save Results**: Models are automatically saved to `/kaggle/working/`

## 📊 Expected Results

All models should achieve:
- **Training Accuracy**: 80-95%
- **Validation Accuracy**: 70-85%
- **Test Accuracy**: 65-80%

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've cloned the repository correctly and updated the GitHub URL
   - The notebooks now include debugging output to help identify path issues
   - If the normal import fails, the notebooks will automatically try a direct file import
2. **CUDA Out of Memory**: Reduce batch size in the training configuration
3. **Dataset Not Found**: Ensure ESC-50 is uploaded to your Kaggle workspace and accessible
4. **Mamba Installation Issues**: Use the exact installation commands in the Mamba notebook
5. **Repository Clone Issues**: Make sure your GitHub repository is public or you have proper access
6. **Module Not Found**: The notebooks now include fallback import mechanisms that should handle most import issues

### Memory Optimization

- **Liquid S4**: Reduce `d_model` and `n_layers` if needed
- **Mamba**: Reduce `d_model` and `n_layer` if needed
- **VJEPA2**: Reduce `embed_dim` and `depth` if needed

## 📈 Model Comparison

| Model | Parameters | Memory | Speed | Accuracy |
|-------|------------|--------|-------|----------|
| Liquid S4 | ~2M | Low | Fast | Good |
| Mamba | ~8M | Medium | Medium | Very Good |
| VJEPA2 | ~15M | High | Slow | Excellent |

## 🎯 Next Steps

After training:
1. Compare results across models
2. Experiment with hyperparameters
3. Try different data augmentations
4. Fine-tune on your specific audio tasks

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all file paths are correct
3. Ensure you have sufficient GPU memory
4. Check that all dependencies are installed correctly

---

**Happy Training! 🎵🤖**
