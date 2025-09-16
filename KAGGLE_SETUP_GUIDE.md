# 🎯 Kaggle Setup Guide - FREE GPU Training!

## 🎉 Why Kaggle is Perfect for Your Project

**✅ Completely FREE** - No credit card required!  
**✅ Tesla P100 GPU** - 16GB memory (perfect for your models)  
**✅ 30 hours/week** - Resets every Saturday  
**✅ 9-hour sessions** - Plenty of time for training  
**✅ 20GB storage** - More than enough for ESC-50 + models  

## 📊 Your Project vs Kaggle Limits

| Resource | Your Project | Kaggle Limit | Status |
|----------|--------------|--------------|---------|
| **Dataset** | ESC-50 (~1.3GB) | 20GB | ✅ Perfect fit |
| **Model Memory** | 8-16GB | 16GB GPU | ✅ Fits comfortably |
| **Training Time** | 2-5 hours/model | 9 hours/session | ✅ Multiple models per session |
| **Weekly Usage** | ~12 hours total | 30 hours/week | ✅ Can train all models multiple times |

## 🚀 Step-by-Step Setup

### 1. Create Kaggle Account
- Go to [kaggle.com](https://www.kaggle.com/)
- Sign up (completely free)
- Verify your phone number (required for GPU access)

### 2. Start New Notebook
- Click **"Notebooks"** → **"New Notebook"**
- Choose **"Code"** (not "Data" or "Models")
- In settings (⚙️), set **Accelerator** to **"GPU T4 x2"** or **"GPU P100"**

### 3. Upload Your Notebook
- Upload the `kaggle_setup.ipynb` file I created
- Or create a new notebook and copy the cells

### 4. Update GitHub URL
- In cell 3, replace `YOUR_USERNAME` with your actual GitHub username
- Example: `!git clone https://github.com/johnsmith/audio-classifier.git`

### 5. Run Training
- Execute cells 1-5 for setup
- Choose one training cell (6-8) to run your model
- Download results when complete

## 🎯 Training Strategy

### Option 1: Single Session (Recommended)
**Train Mamba first** (fastest, ~2-3 hours):
```python
# Run cell 7 - Train Mamba Model
```

### Option 2: Multiple Sessions
**Week 1**: Train Mamba (2-3 hours)  
**Week 2**: Train Liquid S4 (3-4 hours)  
**Week 3**: Train V-JEPA2 (4-5 hours)  

### Option 3: All Models (If you have time)
Train all three models in separate sessions within your 30-hour weekly limit.

## 💡 Kaggle-Specific Optimizations

### Memory Management
- **Batch sizes**: Automatically optimized for Kaggle's GPU
- **Mixed precision**: Enabled by default in our cloud training script
- **Gradient accumulation**: Used if needed for larger effective batch sizes

### Session Management
- **Keep-alive**: Kaggle has 20-minute idle timeout
- **Checkpointing**: Models saved every epoch to prevent data loss
- **Download results**: Always download checkpoints before session ends

### Storage Optimization
- **Dataset**: Downloaded once, cached for future sessions
- **Models**: Only final checkpoints saved (not intermediate)
- **Logs**: Compressed and downloaded as zip files

## 🔧 Troubleshooting

### Common Issues:

**"CUDA out of memory"**
```python
# Reduce batch size
!python scripts/cloud_train.py --model mamba --batch_size 8
```

**"Session timeout"**
- Kaggle sessions timeout after 20 minutes of inactivity
- Keep the notebook active or implement keep-alive

**"Git clone fails"**
- Make sure your repo is public or use GitHub token
- Check the URL in cell 3

**"Import errors"**
- Run cell 2 to install missing packages
- Check that all submodules are initialized

### Performance Tips:

1. **Start with Mamba** - Fastest training, good results
2. **Use smaller batch sizes** if memory is limited
3. **Enable mixed precision** for 1.5x speedup
4. **Monitor GPU usage** with `nvidia-smi` if available

## 📈 Expected Results

### Training Times (Kaggle P100):
- **Mamba**: 2-3 hours → ~85-90% accuracy
- **Liquid S4**: 3-4 hours → ~80-85% accuracy  
- **V-JEPA2**: 4-5 hours → ~85-90% accuracy

### Resource Usage:
- **GPU Memory**: 8-16GB (well within 16GB limit)
- **Storage**: ~3-5GB total (dataset + models + logs)
- **Time**: 9-12 hours total for all models

## 🎉 Success Checklist

- [ ] Kaggle account created and verified
- [ ] GPU enabled in notebook settings
- [ ] Repository cloned successfully
- [ ] ESC-50 dataset downloaded
- [ ] All models tested and created
- [ ] Training completed successfully
- [ ] Checkpoints downloaded
- [ ] Results analyzed

## 🔄 Next Steps After Training

1. **Download models** from the final cell
2. **Compare results** between Mamba, Liquid S4, and V-JEPA2
3. **Test inference** on new audio samples
4. **Share results** on Kaggle or GitHub
5. **Train additional models** next week (30 hours reset!)

## 💰 Cost Comparison

| Platform | Cost | GPU | Training Time | Best For |
|----------|------|-----|---------------|----------|
| **Kaggle** | **FREE** | P100 (16GB) | 2-5 hours | **Your project!** |
| Google Colab | $10-50/month | T4/V100 | 2-5 hours | Quick testing |
| AWS SageMaker | $1-5/hour | Various | 2-5 hours | Production |

**Winner: Kaggle is perfect for your project!** 🏆

---

**Ready to start?** Upload `kaggle_setup.ipynb` to Kaggle and begin training! 🚀
