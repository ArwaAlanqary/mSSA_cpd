function [ WINDOWS ] = sliding_window( X, windowSize, step )


[num_dims, num_samples] = size(X);

WINDOWS = zeros(num_dims * windowSize * step , ...
    floor(num_samples/windowSize/step));

for i = 1 : step : num_samples
    offset = windowSize * step;
    if i + offset -1 > num_samples
        break;
    end
    w = X(:, i: i+ offset -1)';
    WINDOWS(:, ceil(i/step) ) = w(:);
end

end

