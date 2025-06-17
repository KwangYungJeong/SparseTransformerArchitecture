# Sparse Transformer Attention Masks

This project implements and visualizes sparse attention patterns for transformer models. Sparse attention patterns significantly reduce computational complexity and memory usage while maintaining the model's ability to capture both local and global dependencies. While CIFAR-10 (32x32 RGB images) is used as an example, the implementation is general and can be applied to various sequence lengths and tasks.

**한국어 버전은 [README_kr.md](README_kr.md)를 참고하세요.**

## Scripts Overview

This project contains two main scripts:

1. `sparse_transformer_mask.py`: Implements and visualizes different attention patterns
2. `mask_resource_calculator.py`: Analyzes resource requirements for mask operations

### 1. Sparse Transformer Mask Generator

The `sparse_transformer_mask.py` script focuses on generating and visualizing different attention patterns.

```bash
python sparse_transformer_mask.py --help
usage: sparse_transformer_mask.py [-h] [--size SIZE] [--window_size WINDOW_SIZE] [--stride STRIDE] 
                                 [--sample_size SAMPLE_SIZE] [--full_png] [--no_graphic] 
                                 [--order {row_first,column_first}] [--skip_pattern_print]

Sparse Transformer Attention Mask Implementation

options:
  -h, --help            show this help message and exit
  --size SIZE           Size of the square mask matrix (default: 1024)
  --window_size WINDOW_SIZE
                        Size of the local attention window (default: 32)
  --stride STRIDE       Stride between attention points (default: 32)
  --sample_size SAMPLE_SIZE
                        Size of the sample to visualize (default: 128)
  --full_png            Save full-size PNG images instead of samples
  --no_graphic          Bypass graphics and only print sparsity patterns
  --order {row_first,column_first}
                        Order of non-zero elements in the mask (default: row_first)
  --skip_pattern_print  Skip printing non-zero elements of the mask
```

### 2. Mask Resource Calculator

The `mask_resource_calculator.py` script analyzes the resource requirements for mask operations, including:
- Number of unique rows and columns needed
- Changes in required resources between consecutive operations
- Maximum resource usage cases

```bash
python mask_resource_calculator.py --help
usage: mask_resource_calculator.py [-h] [--mask_size MASK_SIZE] [--num_multiplications NUM_MULTIPLICATIONS]
                                   [--window_size WINDOW_SIZE] [--stride STRIDE] [--read_limit READ_LIMIT]
                                   [--zigzag]

Calculate mask resources for sparse matrix multiplications.

options:
  -h, --help            show this help message and exit
  --mask_size MASK_SIZE
                        Size of the square mask matrix (default: 1024)
  --num_multiplications NUM_MULTIPLICATIONS
                        Total number of simultaneous multiplications (default: 64)
  --window_size WINDOW_SIZE
                        Size of the local attention window for strided/fixed masks (default: 32)
  --stride STRIDE       Stride for strided attention (default: 32)
  --read_limit READ_LIMIT
                        Limit the number of lines read from the file (default: 1000)
  --zigzag             Enable zigzag pattern for column sorting in odd rows (default: False)
```

The script generates two types of files for each mask type in the `generated` directory:
1. `{mask_type}_mask_{num_multiplications}_read_limit_{read_limit}[_zigzag].txt`: Contains the calculation points
2. `{mask_type}_mask_{num_multiplications}_read_limit_{read_limit}[_zigzag]_analysis.txt`: Contains the analysis results

The `_zigzag` suffix is added to filenames when the zigzag pattern is enabled.

## Attention Patterns

Five types of attention patterns are implemented:

### 1. Normal (Full) Attention

The normal attention pattern implements standard causal attention:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Lower triangular attention (value 2)**: Each token attends to all previous tokens

This pattern has **49.6% sparsity** (only upper triangular part is masked out).

![Normal Attention](images/normal_mask_128x128.png)

### 2. Strided Pattern

The strided pattern combines three types of attention:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Local window attention (value 2)**: Each token attends to nearby tokens within a local window
- **Strided attention (value 3)**: Each token attends to previous tokens at fixed intervals

This pattern achieves approximately **95.4% sparsity** with a window size and stride of 32.

![Strided Pattern](images/strided_mask_128x128.png)

### 3. Fixed Pattern

The fixed pattern uses:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Block-wise local attention (value 2)**: Each token attends to previous tokens within the same block
- **Fixed column attention (value 3)**: Each token attends to the last token of each previous block

This pattern achieves approximately **96.9% sparsity** with a block size of 32.

![Fixed Pattern](images/fixed_mask_128x128.png)

### 4. Sliding Window Pattern

Each token attends to a fixed-size window of neighboring tokens (excluding itself):
- **Diagonal self-attention (value 1)**
- **Sliding window attention (value 2)**

This pattern achieves approximately **96.8% sparsity** with a window size of 32.

![Sliding Window Pattern](images/sliding_window_mask_128x128.png)

### 5. Dilated Sliding Window Pattern

Each token attends to a fixed number of positions spaced by a dilation factor:
- **Diagonal self-attention (value 1)**
- **Dilated sliding window attention (value 2)**

This pattern achieves approximately **96.8% sparsity** with a window size of 32 and dilation of 2.

![Dilated Sliding Window Pattern](images/dilated_sliding_window_mask_128x128.png)

## Hardware Simulation and Analysis (`spt/` directory)

The `spt/` directory contains scripts and generated plots for simulating and analyzing the performance of sparse attention patterns on a Processing Element (PE) array, a common hardware architecture for accelerating matrix multiplications in deep learning. These simulations evaluate different dataflow strategies (row-wise, column-wise) and provide insights into cycle counts and PE utilization.

### Key Simulation Scripts:
- `spt_mask.py`: Provides mask generation functions, similar to the main `sparse_transformer_mask.py`, used within the simulation environment.
- `spt_sim.py`: A basic PE array simulator for sparse attention.
- `spt_row.py` & `spt_col.py`: Implement more detailed row-wise and column-wise dataflow simulations, including memory (DRAM/SRAM) and DMA modeling.
- `spt_sim4.py`: A comprehensive simulation script that runs various configurations (different mask types, sizes, embedding dimensions) and generates performance analysis plots.

### Performance Analysis Plots (`spt/plots/`):
The `spt/plots/` subdirectory contains numerous visualizations generated by the simulation scripts, illustrating hardware performance metrics. These plots are available for various matrix sizes (e.g., 16x16, 32x32, 64x64) and embedding dimensions (e.g., 16, 32, 64).

- **Cycle Comparison Plots (`cycle_comp_*.png`)**: Show the total cycles required for different attention patterns.
- **PE Cycle Utilization Plots (`pe_cycle_util_*.png`)**: Illustrate how efficiently Processing Elements are utilized across different stages of computation.
- **Row PE Utilization Plots (`row_pe_util_*.png`)**: Provide insights into PE utilization specifically for row-wise processing.
- **Total Cycle Bar Charts (`*_total_cycle_bar_*.png`)**: Summarize total compute and DRAM access cycles.

#### Example Plots:

![Strided Pattern Cycle Comparison](spt/cycle_comp_strided.png)
*Example: Cycle comparison for Strided Pattern*

![Fixed Pattern PE Cycle Utilization (1024 tokens)](spt/pe_cycle_util_fixed_1024.png)
*Example: PE Cycle Utilization for Fixed Pattern with 1024 tokens*

## Implementation Details

This project is implemented as a single Python file that provides:

- Mask generation functions (step-by-step implementation)
- Mask visualization tools
- Sparsity calculation and statistics
- Side-by-side comparison of different attention patterns

## Usage

### Generating Attention Masks

```python
from sparse_transformer_mask import (
    create_normal_mask_step_by_step,
    create_strided_mask_step_by_step,
    create_fixed_mask_step_by_step,
    create_sliding_window_mask_step_by_step,
    create_dilated_sliding_window_mask_step_by_step
)

# Generate normal (full) attention mask
normal_mask = create_normal_mask_step_by_step(size=1024)

# Generate strided mask
strided_mask = create_strided_mask_step_by_step(size=1024, window_size=32, stride=32)

# Generate fixed mask
fixed_mask = create_fixed_mask_step_by_step(size=1024, window_size=32)

# Generate sliding window mask
sliding_window_mask = create_sliding_window_mask_step_by_step(size=1024, window_size=32)

# Generate dilated sliding window mask
dilated_sliding_window_mask = create_dilated_sliding_window_mask_step_by_step(size=1024, window_size=32, dilation=2)
```

#### Sparsity Information in case of 1024 tokens
```
Sparsity of each mask:
Normal Mask Sparsity: 49.9512%
Strided Mask Sparsity: 95.4086%
Fixed Mask Sparsity: 96.8750%
Sliding Window Mask Sparsity: 96.8033%
Dilated Sliding Window Mask Sparsity: 96.8292%
```

### Converting to Binary Masks

The generated masks have values 1, 2, and 3 to distinguish attention types, but for use in actual transformer models, they should be converted to binary masks (consisting only of 0s and 1s):

```python
import numpy as np
from sparse_transformer_mask import convert_to_binary_mask

# Convert to binary mask (all non-zero values to 1)
binary_strided_mask = convert_to_binary_mask(strided_mask)
binary_fixed_mask = convert_to_binary_mask(fixed_mask)

# Check binary mask sparsity
binary_sparsity = 1.0 - np.count_nonzero(binary_strided_mask) / binary_strided_mask.size
print(f"Binary mask sparsity: {binary_sparsity:.4%}")
```

### Visualizing Masks

```python
from sparse_transformer_mask import visualize_mask_sample
import matplotlib.pyplot as plt

# Mask visualization settings
custom_cmap = plt.cm.colors.ListedColormap(['lightgray', 'darkblue', 'royalblue', 'skyblue'])

# Visualize 64x64 sample
visualize_mask_sample(
    mask=strided_mask,
    title='Strided Pattern Mask',
    sample_size=64,
    colormap=custom_cmap,
    save_path='strided_mask_64x64.png'
)
```

### Comparing Multiple Attention Patterns

```python
from sparse_transformer_mask import visualize_mask_comparison

# Compare all five attention patterns side by side (showing 128x128 samples)
visualize_mask_comparison(
    masks=[
        normal_mask[:128, :128],
        strided_mask[:128, :128],
        fixed_mask[:128, :128],
        sliding_window_mask[:128, :128],
        dilated_sliding_window_mask[:128, :128]
    ],
    titles=[
        'Normal (Full) Attention',
        'Strided Pattern',
        'Fixed Pattern',
        'Sliding Window',
        'Dilated Sliding Window'
    ],
    sample_size=128,
    colormap=custom_cmap,
    save_path='mask_comparison_128x128.png'
)
```

![Comparison of All Patterns](images/mask_comparison_128x128.png)

### Running the Script Directly

```bash
python mask_resource_calculator.py
```
or follow this options

```bash
% python mask_resource_calculator.py --help                           
usage: mask_resource_calculator.py [-h] [--mask_size MASK_SIZE] [--num_multiplications NUM_MULTIPLICATIONS]
                                   [--window_size WINDOW_SIZE] [--stride STRIDE] [--read_limit READ_LIMIT]
                                   [--zigzag]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
