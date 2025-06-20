# mask_resource_calculator.py

import argparse
import numpy as np
import re
from sparse_transformer_mask import (
    create_normal_mask_step_by_step,
    create_strided_mask_step_by_step,
    create_fixed_mask_step_by_step
)

def print_cal_points(mask, num_multiplications, mask_size, read_limit, fname=None, zigzag=False):
    """
    Calculates the number of unique rows and columns needed for the first
    num_multiplications multiplications, considering the read limit.
    
    Args:
        mask: The attention mask matrix
        num_multiplications: Number of simultaneous multiplications
        mask_size: Size of the square mask matrix
        read_limit: Maximum number of unique rows/columns to read
        fname: Optional file name to save results
        zigzag: If True, sort columns in descending order for odd rows (default: False)
    """
    # Get non-zero elements and sort by row, then by column
    non_zero = np.nonzero(mask)
    rows, cols = non_zero
    
    # Create a list of (row, col) pairs
    points = list(zip(rows, cols))
    
    # Sort by row first, then by column
    # For odd rows, sort columns in descending order if zigzag is True
    points.sort(key=lambda x: (x[0], -x[1] if zigzag and x[0] % 2 == 1 else x[1]))
    
    # Open file for writing if fname is provided
    f = open(fname, 'w') if fname else None
    
    try:
        time_step = 0
        current_points = []
        unique_rows = set()
        unique_cols = set()
        
        for row, col in points:
            current_unique_count = len(unique_rows) + len(unique_cols)
            
            # Only check read_limit if we're close to it
            if current_unique_count >= read_limit - 2:
                # Check if adding this point would exceed read_limit
                temp_rows = unique_rows.copy()
                temp_cols = unique_cols.copy()
                temp_rows.add(row)
                temp_cols.add(col)
                
                if len(temp_rows) + len(temp_cols) > read_limit:
                    # Print current batch of points
                    if current_points:
                        point_str = f"at time {time_step}: {', '.join(f'({r}, {c})' for r, c in current_points)}"
                        if f:
                            f.write(point_str + '\n')
                        else:
                            print(point_str)
                        time_step += 1
                    
                    # Reset for next batch
                    current_points = []
                    unique_rows = set()
                    unique_cols = set()
            
            # Add current point
            current_points.append((row, col))
            unique_rows.add(row)
            unique_cols.add(col)
            
            # If we've reached num_multiplications, print and reset
            if len(current_points) == num_multiplications:
                point_str = f"at time {time_step}: {', '.join(f'({r}, {c})' for r, c in current_points)}"
                if f:
                    f.write(point_str + '\n')
                else:
                    print(point_str)
                time_step += 1
                current_points = []
                unique_rows = set()
                unique_cols = set()
        
        # Print any remaining points
        if current_points:
            point_str = f"at time {time_step}: {', '.join(f'({r}, {c})' for r, c in current_points)}"
            if f:
                f.write(point_str + '\n')
            else:
                print(point_str)
                
    finally:
        if f:
            f.close()
    
    return len(unique_rows), len(unique_cols)

def write_parameters_to_file(fname, MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, read_limit, type = "normal"):
    with open(fname, 'w') as f:
        f.write(f"# Type: {type}\n")
        f.write(f"# Mask Size: {MASK_SIZE}\n")
        f.write(f"# Number of Multiplications: {NUM_MULTIPLICATIONS}\n")
        f.write(f"# Window Size: {WINDOW_SIZE}\n")
        f.write(f"# Stride: {STRIDE}\n")
        f.write(f"# Read Limit: {read_limit}\n")

def analyze_cal_points(ifname, ofname):
    """
    Analyzes the cal points file to extract unique rows and columns.
    Also analyzes the changes between consecutive indices.
    """
    # Track changes between consecutive indices
    max_row_changes = {'time': 0, 'count': 0, 'rows': set()}
    max_col_changes = {'time': 0, 'count': 0, 'cols': set()}
    max_total_changes = {'time': 0, 'count': 0, 'rows': set(), 'cols': set()}
    
    # Track overall maximum case
    max_row_and_col = 0
    max_case_rows = None
    max_case_cols = None
    max_case_points = None
    max_case_time = None
    cal_count = 0

    # Store all points for each time step
    time_points = {}  # {time: [(row, col), ...]}

    with open(ifname, 'r') as f:
        for line in f:
            # get parameters from the # lines
            if line.startswith('#'):
                print(line.strip())
            # Check if the line contains 'at time' to find cal points
            elif line.startswith('at time'):
                cal_count += 1
                # parse the points
                time_str = line.split(':')[0].strip()  # Extract the time part
                time_val = int(time_str.split(' ')[-1])  # Get the time value
                points_string = line.split(': ')[1].strip()
                matches = re.findall(r'\((\d+),\s*(\d+)\)', points_string)
                points_list = [(int(x), int(y)) for x, y in matches]
                
                # Store points for this time step
                time_points[time_val] = points_list

    # Analyze changes between consecutive time steps
    for time in sorted(time_points.keys()):
        if time == 0:  # Skip first time step
            continue
            
        # Get points for current and previous time step
        current_points = time_points[time]
        prev_points = time_points[time - 1]
        
        # Calculate unique rows and columns for current and previous time steps
        current_rows = {row for row, _ in current_points}
        current_cols = {col for _, col in current_points}
        prev_rows = {row for row, _ in prev_points}
        prev_cols = {col for _, col in prev_points}
        
        # Calculate changes
        new_rows = current_rows - prev_rows
        new_cols = current_cols - prev_cols
        total_changes = len(new_rows) + len(new_cols)
        
        # Update max changes if current changes are larger
        if len(new_rows) > max_row_changes['count']:
            max_row_changes = {'time': time, 'count': len(new_rows), 'rows': new_rows}
        if len(new_cols) > max_col_changes['count']:
            max_col_changes = {'time': time, 'count': len(new_cols), 'cols': new_cols}
        if total_changes > max_total_changes['count']:
            max_total_changes = {'time': time, 'count': total_changes, 
                               'rows': new_rows, 'cols': new_cols}
        
        # Update max case if current case has more unique indices
        total_indices = len(current_rows) + len(current_cols)
        if total_indices > max_row_and_col:
            max_row_and_col = total_indices
            max_case_rows = sorted(list(current_rows))
            max_case_cols = sorted(list(current_cols))
            max_case_time = time
            max_case_points = current_points

    # Prepare analysis results
    analysis_results = [
        "\n",
        "--- Analysis Results ---",
        f"Input File: {ifname}",
        f"Total Calculation Points: {cal_count}",
        f"Max Case Time: {max_case_time}",
        f"Unique Rows: {len(max_case_rows)}",
        f"Unique Columns: {len(max_case_cols)}",
        f"Total Unique Indices (Rows + Columns): {max_row_and_col}",
        f"Max Case Row List: {max_case_rows}",
        f"Max Case Column List: {max_case_cols}",
        "\n",
        "\n",
        "--- Change Analysis ---",
        f"Max Row Changes at time {max_row_changes['time']}: {max_row_changes['count']} new rows",
        f"New Rows: {sorted(list(max_row_changes['rows']))}",
        "\n",
        f"Max Column Changes at time {max_col_changes['time']}: {max_col_changes['count']} new columns",
        f"New Columns: {sorted(list(max_col_changes['cols']))}",
        "\n",
        f"Max Total Changes at time {max_total_changes['time']}: {max_total_changes['count']} total changes",
        f"New Rows: {sorted(list(max_total_changes['rows']))}",
        f"New Columns: {sorted(list(max_total_changes['cols']))}"
    ]

    # open a file with prefix from the fname
    output_fp = open(ofname, 'w')

    # Print to screen and write to file
    for line in analysis_results:
        print(line)
        output_fp.write(line + '\n')
    print(f"\nAnalysis results have been saved to: {ofname}")

    output_fp.close()



def _process_mask_type(mask_name, mask_creation_func, mask_creation_args,
                      MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, args_read_limit, args_zigzag=False):
    print("\n" + "*" * 50)
    print(f"--- {mask_name} Mask ---")
    mask = mask_creation_func(*mask_creation_args)
    
    # Create base filename
    base_filename = f"generated/{mask_name.lower().replace(' ', '_')}_mask_{NUM_MULTIPLICATIONS}_read_limit_{args_read_limit}"
    
    # Add _zigzag suffix if zigzag is True
    if args_zigzag:
        base_filename += "_zigzag"
    
    points_filename = f"{base_filename}.txt"
    analysis_filename = f"{base_filename}_analysis.txt"

    write_parameters_to_file(points_filename, MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, args_read_limit, type=mask_name.lower().replace(' ', '_'))
    
    print_cal_points(mask, NUM_MULTIPLICATIONS, MASK_SIZE, read_limit=args_read_limit, fname=points_filename, zigzag=args_zigzag)
    analyze_cal_points(points_filename, analysis_filename)

def main():
    parser = argparse.ArgumentParser(description='Calculate mask resources for sparse matrix multiplications.')
    parser.add_argument('--mask_size', type=int, default=1024,
                        help='Size of the square mask matrix (default: 1024)')
    parser.add_argument('--num_multiplications', type=int, default=64,
                        help='Total number of simultaneous multiplications (default: 64)')
    parser.add_argument('--window_size', type=int, default=32,
                        help='Size of the local attention window for strided/fixed masks (default: 32)')
    parser.add_argument('--stride', type=int, default=32,
                        help='Stride for strided attention (default: 32)')
    parser.add_argument('--read_limit', type=int, default=1000,
                        help='Limit the number of lines read from the file (default: 1000)')
    parser.add_argument('--zigzag', action='store_true',
                        help='Sort columns in descending order for odd rows (default: False)')
    
    args = parser.parse_args()
    
    MASK_SIZE = args.mask_size
    NUM_MULTIPLICATIONS = args.num_multiplications
    WINDOW_SIZE = args.window_size
    STRIDE = args.stride

    print("--- Input Parameters ---")
    print(f"Mask Size: {MASK_SIZE}")
    print(f"Number of Multiplications: {NUM_MULTIPLICATIONS}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Read Limit: {args.read_limit}")
    print(f"zigzag: {args.zigzag}\n")

    print("*" * 50)
    print(f"Calculating worst-case resources for {NUM_MULTIPLICATIONS} multiplications with MASK_SIZE={MASK_SIZE}:\n")

    # Normal Mask
    _process_mask_type("Normal", create_normal_mask_step_by_step, (MASK_SIZE,),
                      MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, args.read_limit, args.zigzag)

    # Strided Mask
    _process_mask_type("Strided", create_strided_mask_step_by_step, (MASK_SIZE, WINDOW_SIZE, STRIDE),
                      MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, args.read_limit, args.zigzag)

    # Fixed Mask
    _process_mask_type("Fixed", create_fixed_mask_step_by_step, (MASK_SIZE, WINDOW_SIZE),
                      MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, args.read_limit, args.zigzag)

    print("\n" + "*" * 50)
    print("Resource calculation completed.")
    print("Check the generated files in the 'generated' directory for detailed results.")

if __name__ == "__main__":
    main()
