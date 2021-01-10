clear;
%%
figure
load logwell.mat
% choice of alpha
alpha = .01;

n = 50;
k = 10;
subplot(2,1,1)
plot(y')
axis([1,size(y,2),-inf,inf])
%%
score1 = change_detection(y,n,k,alpha,5);
score2 = change_detection(y(:,end:-1:1),n,k,alpha,5);
%%
subplot(2,1,1);
plot(y', 'linewidth',2);
axis([-inf,size(y,2),-inf,inf])
title('Original Signal')
%%
subplot(2,1,2);
score2 = score2(end:-1:1);

% 2*n-2+k is the size of buffer zone
plot([zeros(1,2*n-2+k),score1 + score2], 'r-', 'linewidth',2);
axis([-inf,size(y,2),-inf,inf])
title('Change-Point Score')
