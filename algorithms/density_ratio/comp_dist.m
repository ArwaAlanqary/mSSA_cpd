function [dist2] = comp_dist(x,y)
[d, nx] = size(x);
[d, ny] = size(y);

G = sum(x.*x,1);
T = repmat(G,ny,1);
G = sum(y.*y,1);
R = repmat(G,nx,1);

dist2 = T' + R - 2.*x'*y;
end

