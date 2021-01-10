function [med] = comp_med(x)

[d,n] = size(x);
G = sum(x.*x,1);
T = repmat(G,n,1);
dist2 = T - 2*x'*x + T';
dist2 = dist2 - tril(dist2);
R = dist2(:);
med = sqrt(.5 * median(R(R>0))); % rbf kernel has 2

end
