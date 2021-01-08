function save_results_table(experiment, score, path, status)
    tabel_results = [experiment.algorithm_name, experiment.dataset, ...
               experiment.data_name, status, score];
    dlmwrite(fullfile(path, 'results_table.csv'),tabel_results,...
            'delimiter',',','-append');




end 