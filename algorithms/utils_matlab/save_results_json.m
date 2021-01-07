function save_results_json(experiment, param, cp, score, path, ...
                           status, error)
    results = struct;
    results.status = status;
    results.error = error;
    results.algorithm = experiment.algorithm_name;
    results.dataset = experiment.dataset;
    results.data_name = experiment.data_name;
    results.param = param;
    if status == 'success'
        results.cp = cp;
        results.score = struct('metric', experiment.metric, 'value', score);
    else
        results.cp = NaN;
        results.score = NaN;
    end
    file_name = experiment.data_name + "_" + ...
                string(java.util.UUID.randomUUID) + ".json";
    results_json_text = jsonencode(results);
    
    file = fopen(fullfile(path, file_name), 'w');
    fwrite(file, results_json_text, 'char');
    fclose(file);  
end
