import time
import torch

def profile_training(model, data, target, optimizer, criterion):
    """Profile different parts of the training process to identify bottlenecks."""
    times = {}
    
    # 1. Data transfer time (if needed)
    if data.device != model.device:
        start = time.time()
        data = data.to(model.device)
        target = target.to(model.device)
        times['data_transfer'] = time.time() - start
    
    # 2. Forward pass
    start = time.time()
    output = model(data)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    times['forward'] = time.time() - start
    
    # 3. Loss calculation
    start = time.time()
    loss, metrics = criterion(output, target)
    torch.cuda.synchronize()
    times['loss'] = time.time() - start
    
    # 4. Backward pass
    start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    times['backward'] = time.time() - start
    
    # 5. Optimizer step
    start = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    times['optimizer'] = time.time() - start
    
    # 6. Memory info
    if torch.cuda.is_available():
        times['memory'] = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
    
    return times, output
