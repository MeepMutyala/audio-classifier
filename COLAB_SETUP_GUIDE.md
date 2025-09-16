# 🚀 Colab Setup Guide

## Quick Start (Git-based approach - Recommended)

### 1. Push Your Code to GitHub

```bash
# If you haven't initialized git yet
git init
git remote add origin https://github.com/YOUR_USERNAME/audio-classifier.git

# Add and commit all files
git add .
git commit -m "Add cloud training support and Colab notebook"

# Push to GitHub
git push -u origin main
```

### 2. Set Up Colab

1. **Go to Google Colab**: https://colab.research.google.com/
2. **Upload the notebook**: Upload `colab_setup.ipynb` 
3. **Update the GitHub URL**: In cell 2, replace `YOUR_USERNAME` with your actual GitHub username
4. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or V100)
5. **Run all cells**: Execute cells 1-4 to setup, then choose a training cell (5-7)

## What Gets Cloned

The Colab notebook will automatically:
- ✅ Clone your entire repository with all submodules
- ✅ Download the ESC-50 dataset (~1.3GB)
- ✅ Install all required dependencies
- ✅ Test all three models (Mamba, Liquid S4, V-JEPA2)
- ✅ Run training with cloud-optimized settings
- ✅ Download trained models when complete

## Training Options

| Model | Training Time | GPU Memory | Best For |
|-------|---------------|------------|----------|
| **Mamba** | ~2-3 hours | 8GB | Quick results |
| **Liquid S4** | ~3-4 hours | 12GB | Efficient training |
| **V-JEPA2** | ~4-5 hours | 16GB | Best accuracy |

## Alternative: Manual Upload (If Git doesn't work)

If you prefer to upload files manually:

### Files to Upload to Colab:
```
audio-classifier/
├── src/                    # All source code
├── external_models/        # Mamba, Liquid S4, V-JEPA2 submodules
├── configs/               # Model configurations
├── scripts/               # Training scripts
├── data/ESC-50/          # Dataset (or let Colab download it)
├── requirements-cloud.txt # Dependencies
└── colab_setup.ipynb     # The notebook
```

### Manual Setup Steps:
1. Upload all files to Colab
2. Run: `!pip install -r requirements-cloud.txt`
3. Run: `!git submodule update --init --recursive`
4. Run training: `!python scripts/cloud_train.py --model mamba`

## Troubleshooting

### Common Issues:
- **"No module named 'src'"**: Make sure you're in the project directory
- **"CUDA out of memory"**: Reduce batch size in training commands
- **"Git submodule errors"**: Run `!git submodule update --init --recursive`

### Memory Optimization:
- Use smaller batch sizes: `--batch_size 8` or `--batch_size 4`
- Enable mixed precision: `--mixed_precision`
- Use the cloud-optimized script: `scripts/cloud_train.py`

## Expected Results

After training, you'll get:
- 📁 `checkpoints/` - Trained model weights
- 📊 Training logs with accuracy/loss curves
- 🎯 Model performance on ESC-50 test set
- 📈 Weights & Biases integration (if configured)

## Next Steps

1. **Start with Mamba** - Fastest to train and test
2. **Compare results** - Train all three models
3. **Scale up** - Move to AWS SageMaker for production
4. **Deploy** - Use trained models for inference

Happy training! 🎵🤖
