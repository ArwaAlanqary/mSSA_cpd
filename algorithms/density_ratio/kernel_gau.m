function [k] = kernel_gau(dist2, sigma)
k = exp(-dist2/(2*sigma^2));
end