function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% X = m x n
%initial_centroids = [3 3; 6 2; 8 5];

m = size(X, 1);

distance = zeros(m,K);



  for j=1:K
    diff_col = X-centroids(j,:);
    power_col = diff_col.^2;
    distance(:,j) = sum(power_col,2);
    end
 [value,idx] = min(distance,[],2);
 
  
  
%for i=1:m
%  for j=1:K
%    distance = (X(i,:)-centroids(j,:));
%    distance = sum(distance.^2);
%
%    if j==1
%      minIndex = 1;
%      minDistance = distance;
%    end
%
%    if distance < minDistance
%      minDistance = distance;
%      minIndex = j;
%    end
%  end
%
% idx(i) = minIndex; 
%end
  
  
  
  
  
  
  
  
    
    
% =============================================================

end

