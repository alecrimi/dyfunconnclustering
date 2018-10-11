
% Load ASD files
list_ASD = dir('ASD*.*');
list_control = dir('control*.*');

maxClusters = 25;
n_window = 5; 
n_parcels = 96;
tot_sample = length(list_ASD) + length(list_control);
res_cla= zeros(n_parcels,tot_sample); 

tot_data = zeros(n_parcels,n_parcels,tot_sample);
K = zeros(tot_sample,1);

for kk = 1 : length(list_ASD)
       temp = csvread(list_ASD(kk).name);
       temp( find(temp<0) )= 0;
       tot_data(:,:,kk) = temp;
end


for jj = 1 : length(list_control)
       temp = csvread(list_control(jj).name);
       
       temp( find(temp<0) )= 0;
       tot_data(:,:,jj+kk) = temp;
       
end


gamma  = 1 ;
omega = 0.1;
N=n_parcels;%length(A{1});
T=n_window;% length(A);
B=spalloc(N*T,N*T,N*N*T+2*N*T);

F_tot = zeros(n_parcels,tot_sample/n_window);

for xx = 1 : n_window : tot_sample
        % Convert matrix to cell
        A = { tot_data(:,:,xx ) tot_data(:,:,xx+1 ) tot_data(:,:,xx+2 ) tot_data(:,:,xx+3 ) tot_data(:,:,xx+4) };
        % xx
	twomu=0;
	for s=1:T
	    k=sum(A{s});
	    twom=sum(k);
	    twomu=twomu+twom;
	    indx=[1:N]+(s-1)*N;
	    B(indx,indx)=A{s}-gamma*k'*k/twom;
	end
	twomu=twomu+2*omega*N*(T-1);
	B = B + omega*spdiags(ones(N*T,2),[-N,N],N*T,N*T);
    
	%[S,Q] = genlouvain(B);%(B,10000,0,0);
	%Q = Q/twomu; 
        [S,Q,n_it] = iterated_genlouvain(B,10000,0,0,'move',[], []);
        S = reshape(S, N, T);
    
        F_tot(:,xx) = flexibility(S'); 
end

F_m = F_tot(:,1:n_window:end);
F_ASD = F_m(:,1:length(list_ASD)/n_window);
F_control = F_m(:,length(list_ASD)/n_window+1:end);

[h,p] = ttest2(F_ASD',F_control');
val = p < 0.05;

[h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(p ,0.0001,'pdep','yes');
val = adj_p < 0.0001;
