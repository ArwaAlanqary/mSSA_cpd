function f1_score = compute_f1_score(actual, detected, margin)  
    actual_cp = actual(:, 1); 
    detected_cp = detected(:, 1); 
    
    %compute true positive
    true_positive = 0;
    for cp = transpose(actual_cp)
        if min(abs(detected_cp - cp)) <= margin
            true_positive = true_positive + 1;
        end   
    end
    
    %compute recall
    actual_cp_number = length(actual);
    recall  = true_positive/actual_cp_number;
    
    %compute precision
    total_number_of_detected_cp = length(detected_cp);
    precision = true_positive/total_number_of_detected_cp;
    
    %compute f1 socre
    f1_score = 2*recall*precision/(recall+precision);
   
end