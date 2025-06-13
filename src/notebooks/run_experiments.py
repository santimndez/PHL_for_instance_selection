import argparse
import csv
import subprocess
import sys
import time

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Batch runner for experiment.py using parameters from a CSV file.\n"
                    "The CSV must contain the columns: dataset, output, and target."
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='./experiments.csv',
        help='Path to the input CSV file with experiment configurations. Default: ./experiments.csv.'
    )
    parser.add_argument(
        '--profiling',
        action='store_true',
        help='Enable profiling mode for all experiments (adds --profiling flag to each run).'
    )
    args = parser.parse_args()

    try:
        # Open and read the CSV file
        with open(args.csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Read parameters from the current row
                dataset = row['dataset']
                output = row['output']
                target = row['target']

                # Build the command to run experiment.py
                command = [
                    'python', 'experiment.py',
                    '-d', dataset,
                    '-o', output,
                    '-t', target,
                    '--seed', '2025'
                ]

                # Optionally add --profiling
                if args.profiling:
                    command.append('--profiling')

                print(f"Running: {' '.join(command)}")
                t0 = time.time()
                subprocess.run(command)
                elapsed_time = time.time() - t0
                print(f"Finished in {elapsed_time:.2f} seconds")
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing column in CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
