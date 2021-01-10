function [rPE, g_nu, g_re, sigma_chosen, lambda_chosen] = RelULSIF( x_de, x_nu, x_re, x_ce, alpha, fold)
rng default

if nargin < 6 || isempty(fold)
    fold = 5;
end
[~,n_nu] = size(x_nu);
[~,n_de] = size(x_de);

% Parameter Initialization Section
if nargin < 4 || isempty(x_ce)
    b = min(100,n_nu);
    idx = randperm(n_nu);
    x_ce = x_nu(:,idx(1:b));
end

if nargin < 5
    alpha = 0.5;
end

% construct gaussian centers
[~,n_ce] = size(x_ce);
% get sigma candidates
x = [x_de,x_nu];
med = comp_med(x);
sigma_list = med * [.6,.8,1.0,1.2,1.4];
% get lambda candidates
lambda_list = 10.^[-3:1:1];

[dist2_de] = comp_dist(x_de,x_ce);
%n_de * n_ce
[dist2_nu] = comp_dist(x_nu,x_ce);
%n_nu * n_ce

%The Cross validation Section Begins
score = zeros(length(sigma_list),length(lambda_list));
for i = 1:length(sigma_list)
    k_de = kernel_gau(dist2_de,sigma_list(i));
    k_nu = kernel_gau(dist2_nu,sigma_list(i));
    for j = 1:length(lambda_list)
        
        cv_index_nu=randperm(n_nu);
        cv_split_nu=floor([0:n_nu-1]*fold./n_nu)+1;
        cv_index_de=randperm(n_de);
        cv_split_de=floor([0:n_de-1]*fold./n_de)+1;
        
        sum = 0;
        for k = 1:fold
            k_de_k = k_de(cv_index_de(cv_split_de~=k),:)';
            %n_ce * n_de
            k_nu_k = k_nu(cv_index_nu(cv_split_nu~=k),:)';
            %n_ce * n_nu
            
            H_k = ((1-alpha)/size(k_de_k,2))*k_de_k*k_de_k' + ...
                (alpha/size(k_nu_k,2))*k_nu_k*k_nu_k';
            h_k = mean(k_nu_k,2);
            
            theta = (H_k + eye(n_ce)*lambda_list(j))\h_k;
            %theta = max(theta,0);
            
            k_de_test = k_de(cv_index_de(cv_split_de==k),:)';
            k_nu_test = k_nu(cv_index_nu(cv_split_nu==k),:)';
            % objective function value
            J = alpha/2 * mean((theta' * k_nu_test).^2)+ ...
                (1-alpha)/2*mean((theta'*k_de_test).^2)- ...
                mean(theta' * k_nu_test);
            sum = sum + J;
        end
        score(i,j) = sum/fold;
    end
end

%find the chosen sigma and lambda
[i_min,j_min] = find(score==min(score(:)));
sigma_chosen = sigma_list(i_min);
lambda_chosen = lambda_list(j_min);

%compute the final result
k_de = kernel_gau(dist2_de',sigma_chosen);
k_nu = kernel_gau(dist2_nu',sigma_chosen);

H = ((1-alpha)/n_de)*k_de*k_de' + ...
    (alpha/n_nu)*k_nu*k_nu';
h = mean(k_nu,2);

theta = (H + eye(n_ce)*lambda_chosen)\h;

g_nu = theta'*k_nu;
g_de = theta'*k_de;
g_re = [];
if ~isempty(x_re)
    dist2_re = comp_dist(x_re,x_ce);
    k_re = kernel_gau(dist2_re', sigma_chosen);
    g_re = theta'*k_re;
end
rPE = mean(g_nu) - 1/2*(alpha*mean(g_nu.^2) + ...
    (1-alpha)*mean(g_de.^2)) - 1/2;

end