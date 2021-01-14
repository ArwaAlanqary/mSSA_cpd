function [SCORE, sigma_track, lambda_track] = change_detection(X, n, k, alpha, fold)

SCORE=[];

WIN = sliding_window(X, k, 1);
nSamples = size(WIN,2);
t = n +1;
sigma_track = [];
lambda_track = [];
while(t + n -1 <= nSamples)
    
    Y = [WIN(:, t-n : n + t -1 )];
    Y = Y./repmat(std(Y,0,2),1,2*n);
    YRef = Y(:,1:n);
    YTest = Y(:,n+1:end);
    
    [s,~,~,sig, lam]=RelULSIF(YRef,YTest,[],[],alpha, fold);
    sigma_track = [sigma_track, sig];
    lambda_track = [lambda_track, lam];
    
    %print out the progress
%     if(mod(t,20) == 0)
%         display(sprintf('. %d',t))
%     end
    
    SCORE = [SCORE, s];
    t = t + 1;
end

end

