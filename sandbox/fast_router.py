import torch
from torch.utils.cpp_extension import load
import time
import os

print("\n===========================================================")
print("[INIT] Compiling raw C++ Kernel Backend using JIT Compiler...")
print("       (Looking for System C++ Compilers...)")
print("===========================================================\n")

# Locate the C++ mathematically optimized file
source_file = os.path.join(os.path.dirname(__file__), 'csrc', 'router_kernel.cpp')

try:
    print("[STATUS] Attempting to hook into Microsoft Visual C++ Build Tools...")
    
    # Compile the C++ extension dynamically deep into PyTorch
    fast_backend = load(
        name="router_kernel",
        sources=[source_file],
        verbose=True
    )
    print("\n[SUCCESS] C++ Source successfully compiled into Python backend executable!")

    # Setup the physical Matrix limits
    d_model = 64
    num_hyperplanes = 4 # 2^4 = 16 available Expert Latent Vector pools
    batch_size = 128
    seq_len = 100

    print("[STATUS] Loading massive randomized token blocks...")
    context = torch.randn(batch_size, seq_len, d_model)
    fixed_hyperplanes = torch.randn(d_model, num_hyperplanes)

    # Execute math directly through the C++ Matrix function
    print("\n[EXEC] Passing Semantic Vectors through C++ SparseLSHGate...")
    start = time.time()
    
    routes = fast_backend.sparse_route(context, fixed_hyperplanes)
    
    end = time.time()

    print(f"\n[RESULT] Native C++ Latent Output Map produced in: {(end - start):.5f} seconds!")
    unique_pools = len(torch.unique(routes))
    print(f"         Unique LatentExpertPools mathematically targeted: {unique_pools}/16")
    
    print("\n[SUCCESS] The PyTorch <--> C++ backend pipeline is fully functional.")

except Exception as e:
    print("\n-----------------------------------------------------------")
    print("[ERROR] C++ extension build is currently blocked by local toolchain dependencies.")
    print("-----------------------------------------------------------")
    print("This environment is missing one or more build requirements for")
    print("`torch.utils.cpp_extension.load`, such as Ninja or a working compiler toolchain.")
    print("\nError Trace Context:")
    print(str(e)[:500] + "...\n")
