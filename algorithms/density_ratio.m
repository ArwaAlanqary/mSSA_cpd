function [score, cp] = density_ratio(ts, n, k, alpha, thr, peak_dist, fold)
    % ts must be a row vector
    size_ts = size(ts);
    rows = size_ts(1);
    cols = size_ts(2);
    if rows > cols
        ts = transpose(ts);
    end
    
    score1 = change_detection(ts,n,k,alpha,fold);
    score2 = change_detection(ts(:,end:-1:1),n,k,alpha,fold);
    score2 = score2(end:-1:1);
    score = score1+score2;
    threshold = mean(score) + thr*std(score);
    distance = peak_dist*(n+k-2);
    score = [zeros(1,n-2+k),score];
    [~, cp] = findpeaks(score,'MinPeakHeight',threshold, ...
                        'MinPeakDistance',distance);
    if isempty(cp)
        cp = [0, length(ts)-1];
    else  
        if cp(1) ~= 1
            cp = [1, cp];
        end 
        cp = cp - 1;
        cp_end = [cp(2:end)-1, length(ts)-1];
        cp = [transpose(cp), transpose(cp_end)];
    end 

end 