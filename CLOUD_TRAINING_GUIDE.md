# Cloud Training Guide for Audio Classification

## Quick Start Options

### 1. Google Colab (Recommended for Testing)
- **Setup Time**: 5 minutes
- **Cost**: Free tier or $10-50/month
- **Best For**: Quick experiments and testing

**Steps:**
1. Upload `colab_setup.ipynb` to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells to setup and train

### 2. Kaggle Notebooks (Free Alternative)
- **Setup Time**: 10 minutes  
- **Cost**: Free (30 GPU hours/week)
- **Best For**: Free experimentation

**Steps:**
1. Create new Kaggle notebook
2. Upload your code as a dataset
3. Enable GPU in notebook settings
4. Run training scripts

### 3. AWS SageMaker (Production)
- **Setup Time**: 30 minutes
- **Cost**: $0.50-3.00/hour
- **Best For**: Production training

**Steps:**
1. Create SageMaker Studio domain
2. Clone your Git repository
3. Use SageMaker training jobs or Studio notebooks

## Dataset Requirements

Your ESC-50 dataset (~1.3GB) can be:
- **Uploaded directly** to cloud environment
- **Downloaded** during setup (recommended)
- **Stored** in cloud storage (AWS S3, GCP Cloud Storage)

## Model Training Times (Estimated)

| Model | Batch Size | Training Time | GPU Memory |
|-------|------------|---------------|------------|
| Mamba | 32 | ~2-3 hours | 8GB |
| Liquid S4 | 32 | ~3-4 hours | 12GB |
| V-JEPA2 | 16 | ~4-5 hours | 16GB |

## Recommended Cloud Configurations

### Google Colab
- **GPU**: T4 or V100
- **RAM**: 12GB+
- **Storage**: 15GB+

### AWS SageMaker
- **Instance**: ml.g4dn.xlarge (T4 GPU, 4 vCPUs, 16GB RAM)
- **Storage**: 50GB EBS

### Kaggle
- **GPU**: P100 or T4
- **RAM**: 16GB
- **Storage**: 20GB

## Cost Optimization Tips

1. **Use spot instances** when available (50-70% savings)
2. **Monitor training progress** to avoid idle GPU time
3. **Use smaller batch sizes** if memory is limited
4. **Enable mixed precision training** for 1.5x speedup

## Monitoring and Logging

Your code already includes:
- Weights & Biases integration (`wandb`)
- Progress bars with `tqdm`
- Model checkpointing
- Early stopping

Set up W&B account to track experiments across platforms.
