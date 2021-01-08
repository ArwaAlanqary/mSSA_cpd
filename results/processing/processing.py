import pandas as pd
import os
import numpy as np


def generate_average_df(algorithms, data_sets, results_path, output_path): 
    output = {}
    for algorithm in algorithms: 
        output[algorithm] = {}
        for data_set in data_sets: 
            results = pd.read_csv(os.path.join(results_path, algorithm, data_set,'results_table.csv'), header=None)
            results = results[4].values
            output[algorithm][data_set] = np.mean(results)
    output_df = pd.DataFrame.from_dict(output)
    output_df.to_latex(os.path.join(output_path, 'average_f1_table.txt'))






def main():
	data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
	results_path = os.path.join(os.path.dirname(os.getcwd()), 'test')
	output_path = os.path.join(os.path.dirname(os.getcwd()), 'processing')
	algorithms = ['binseg', 'microsoft_ssa', 'hybrid_cusum', 'hybrid_cusum_moving_window']
	data_sets = ['yahoo', 'struct', 'mean', 'energy']
	generate_average_df(algorithms, data_sets, results_path, output_path)
	generate_histograms(algorithms, data_sets, results_path, output_path, data_path)


if __name__ == "__main__":
    main()
