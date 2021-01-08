function cp = detect(ts, n, k, alpha, thr, peak_dist, fold)
    score1 = change_detection(ts,n,k,alpha,fold);
    score2 = change_detection(ts(:,end:-1:1),n,k,alpha,fold);
    score = score1+score2;
    threshold = thr*max(score);
    distance = peak_dist*n;
    [~, cp] = findpeaks(score,'MinPeakHeight',threshold, ...
                        'MinPeakDistance',distance);
end 