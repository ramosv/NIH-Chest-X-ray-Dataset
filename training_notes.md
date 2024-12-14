. **Hardware and Environment Specs:**
   - CPU: Intel i9-14900K (32 threads, max 6.0 GHz)
   - GPU: NVIDIA RTX 4000 Ada Generation (20GB VRAM)
   - RAM: 64GB
   - OS: Ubuntu 24.04.1 LTS
   - CUDA: Version 12.4
   - NVIDIA Driver: 550.120

2. **Pre-run GPU Setup:**
   - Ensure no processes are using the GPU:
     1. Run `nvidia-smi` to check running processes.
     2. Kill any active processes using `kill -9 PID`.
     3. Run `sudo nvidia-smi --gpu-reset` to clear GPU memory.
   - Use `torch.cuda.empty_cache()` in Python to clear allocated memory between runs if required.


3. **Batch Size and Image Size Configurations:**
   - Batch Size: 32
   - Image Size: 224x224
   - Training: 20 epochs
   - Time to completion: 20 minutes

   - Batch Size: 64
   - Image Size: 512x512
   - Training: 20 epochs
   - time to completion: 2 hours

   - Batch Size: 32
   - Image Size: 1024x1024
   - Training: 20 epochs
   - Time to completion: 10 hours

4. **Checkpointing:**
   - Best model is saved as `best_model_size.pth` based on validation loss.
   - Checkpoints are saved every 5 epochs as `checkpoint_epoch_X.pth`.

5. **Performance Observations:**
   - As image size incrases training time increased drastically 

