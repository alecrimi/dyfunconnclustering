
% Load ASD files
list_ASD = dir('SZ*.*');
list_control = dir('CONTROL*.*');

maxClusters = 25;
n_window = 5;

tot_sample = length(list_ASD) + length(list_control);
res_cla= zeros(96,tot_sample); 

tot_data = zeros(96,96,tot_sample);
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
 
for zz = 1 :  tot_sample
        % Compute the degree matrix
        D = zeros(size(tot_data(:,:,zz),1));
        for i=1:size(tot_data(:,:,zz),1)
            D(i,i) = sum(tot_data(i,:,zz));
        end

        % Compute the normalized laplacian / affinity matrix (NL)
        NL = zeros(size(tot_data(:,:,zz)));
        for i=1:size(tot_data(:,:,zz),1)
            for j=1:size(tot_data(:,:,zz),2)
                NL(i,j) = tot_data(i,j,zz) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
            end
        end

        % Remove any nan or inf values in rows or columns
        NL(:,~any(~isnan(NL), 1))=[];
        NL(~any(~isnan(NL), 2),:)=[];
        NL(:,~any(~isinf(NL), 1))=[];
        NL(~any(~isinf(NL), 2),:)=[];

        % Perform the eigenvalue decomposition
        [eigVectors,eigValues] = eig(NL);
        % Select the number of clusters
        gap = zeros(size(NL,1),1);
      zz
        % Maximum number of dynamical clusters

        % Determine number of clusters by finding the maximizer of the eigengap
        for i = 2:size(NL,1)
            if eigValues(i-1,i-1) > 0 && eigValues(i,i) > 0
                gap(i) = abs(eigValues(i-1,i-1) - eigValues(i,i))/eigValues(i,i);
            else
                gap(i) = 0;
            end
        end

        % Choose default number of clusters in the event that maximum of the
        % eigengap is non-unique
        [~,nClusters] = max(gap(1:maxClusters));
        K(zz) = nClusters;
        % Select k largest eigenvectors
        % Choose between the minimum of the maximizer of the eigengap + 10 OR
        % the maximum number of clusters allowed
        % This may mean that it is not feasible/practical to use maxClusters
        %k = min( nClusters, maxClusters);
 
end
k = ceil(mean(K))
%k = 6;
% Now perform all Clustering
for zz = 1 : tot_sample
    
       % k = 6;
% Perform Kmeans clustering on the matrix U
       nEigVec = eigVectors(:,(size(eigVectors,1)-(k-1)): size(eigVectors,1));
       % Construct the normalized matrix U from the obtained eigenvectors
       U = zeros(size(nEigVec));
       for i=1:size(nEigVec,1)
            n = sqrt(sum(nEigVec(i,:).^2));    
            U(i,:) = nEigVec(i,:) ./ n; 
       end
[IDX,C]= kmeans(U,k,'EmptyAction','singleton'); 
res_cla(:,zz) = IDX;
end
% Now perform all Flexibility
F_tot = zeros(96,tot_sample/n_window);
for ll = 1 : n_window : tot_sample
    F_tot(:,ll) = flexibility( res_cla(:,ll:ll+2)' );
end
F_m = F_tot(:,1:n_window:end);
F_ASD = F_m(:,1:length(list_ASD)/n_window);
F_control = F_m(:,length(list_ASD)/n_window+1:end);

[h,p] = ttest2(F_ASD',F_control');
val = p < 0.05;

[h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(p/15,0.05,'pdep','yes');
val = adj_p < 0.05;
