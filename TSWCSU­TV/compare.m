close all
clear all

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of end members
p = 5;  % fixed for this demo

%SNR in dB
SNR = 30;
% noise bandwidth in pixels of the noise  low pass filter (Gaussian)
bandwidth = 1000; % 10000 == iid noise
%bandwidth = 5*pi/224; % colored noise 


% define random states
rand('state',10);
randn('state',10);


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gererate fractional abundances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pure pixels
x1 = eye(p);

% mixtures with two materials
x2 = x1 + circshift(eye(p),[1 0]);

% mixtures with three materials
x3 = x2 + circshift(eye(p),[2 0]);

% mixtures with four  materials
x4 = x3 + circshift(eye(p),[3 0]);

% mixtures with four  materials
x5 = x4 + circshift(eye(p),[4 0]);


% normalize
x2 = x2/2;
x3 = x3/3;
x4 = x4/4;
x5 = x5/5;


% background (random mixture)
%x6 = dirichlet(ones(p,1),1)';
x6 = [0.1149 0.0741  0.2003 0.2055, 0.4051]';   % as in the paper

% build a matrix
xt = [x1 x2 x3 x4 x5 x6];


% build image of indices to xt
imp = zeros(3);
imp(2,2)=1;

imind = [imp*1  imp*2 imp* 3 imp*4 imp*5;
    imp*6  imp*7 imp* 8 imp*9 imp*10;
    imp*11  imp*12 imp*13 imp*14 imp*15;
    imp*16  imp*17 imp* 18 imp*19 imp*20;
    imp*21  imp*22 imp* 23 imp*24 imp*25];

imind = kron(imind,ones(5));

% set backround index
imind(imind == 0) = 26;

% generare frectional abundances for all pixels
[nl,nc] = size(imind);
np = nl*nc;     % number of pixels
for i=1:np
    X(:,i) = xt(:,imind(i));
end

Xim = reshape(X',nl,nc,p);

%  image endmember 1
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% buid the dictionary 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load USGS_1995_Library.mat
%  order bands by increasing wavelength
[dummy index] = sort(datalib(:,1));
A =  datalib(index,4:end);

% prune the library 
% min angle (in degres) between any two signatures 
% the larger min_angle the easier is the sparse regression problem
min_angle = 4.44;       
A = prune_library(A,min_angle); % 240  signature 

% order  the columns of A by decreasing angles 
[A, index, angles] = sort_library_by_angle(A);


%% select p endmembers  from A
%

% angles (a_1,a_j) \simeq min_angle)
supp = 1:p;
M = A(:,supp);
[L,p] = size(M);  % L = number of bands; p = number of material


%%
%---------------------------------
% generate  the observed  data X
%---------------------------------

% set noise standard deviation
sigma = sqrt(sum(sum((M*X).^2))/np/L/10^(SNR/10));
% generate Gaussian iid noise
noise = sigma*randn(L,np);


% make noise correlated by low pass filtering
% low pass filter (Gaussian)
filter_coef = exp(-(0:L-1).^2/2/bandwidth.^2)';
scale = sqrt(L/sum(filter_coef.^2));
filter_coef = scale*filter_coef;
noise = idct(dct(noise).*repmat(filter_coef,1,np));

%  observed spectral vector
Y = M*X + noise;


% create  true X wrt  the library A
n = size(A,2);
N = nl*nc;
XT = zeros(n,N);
XT(supp,:) = X;

for i=1:6
l=[1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];
lambda = l(i);
[X_hat_s] =  sunsal(A,Y,'lambda',lambda,'ADDONE','no','POSITIVITY','yes', ...
'TOL',1e-4, 'AL_iters',2000,'verbose','yes');
SRE_s(i) = 20*log10(norm(XT,'fro')/norm(X_hat_s-XT,'fro'));
end
%CL
for i=1:6
l=[1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];
lambda = l(i);
[X_hat_l21] =  clsunsal_v1(A,Y,'lambda',lambda,'ADDONE','no','POSITIVITY','yes', ...
'TOL',1e-4, 'AL_iters',2000,'verbose','yes');
SRE_l21(i) = 20*log10(norm(XT,'fro')/norm(X_hat_l21-XT,'fro'));
end
% constrained least squares l2-l1-TV (nonisotropic) TV
for i=1:6
for j=1:6
l=[1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];
t=[1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];
lambda  = l(i);
lambda_TV =t(j);
[X_hat_tv_ni,res,rmse_ni] = sunsal_tv(A,Y,'MU',0.01,'POSITIVITY','yes','ADDONE','no', ...
'LAMBDA_1',lambda,'LAMBDA_TV', lambda_TV, 'TV_TYPE','niso',...
'IM_SIZE',[75,75],'AL_ITERS',300, 'TRUE_X', XT,  'VERBOSE','yes');
SRE_tv_ni(i,j) = 20*log10(norm(XT,'fro')/norm(X_hat_tv_ni-XT,'fro'));
end
end
for i=1:6
for j=1:6
l=[1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];
t=[1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];
lambda  = l(i);
lambda_TV =t(j);
[X_hat_ftv,res,rmse_ni] = sunsal_ftv(A,Y,'MU',0.01,'POSITIVITY','yes','ADDONE','no', ...
'LAMBDA_1',lambda,'LAMBDA_TV', lambda_TV, 'TV_TYPE','niso',...
'IM_SIZE',[75,75],'AL_ITERS',300, 'TRUE_X', XT,  'VERBOSE','yes');
SRE_ftv(i,j) = 20*log10(norm(XT,'fro')/norm(X_hat_ftv-XT,'fro'));
end
end