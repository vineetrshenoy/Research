function [accuracy] = simpleNorm(fullMatrix)
%This function takes a full data matrix, creates a 
%testing and training set, and computes a simple norm error,
%It averages the strokes for a user in the test set and computes
%the norm against every vector in the training set
N = 41;

%Obtains our test and training set
[test,train] = test_train_split(fullMatrix,41);


testLabels = test(:,1:2);
test(:,1:2) = [];
test = normr(test);(polar)

trainLabels = train(:,1:2);
train(:,1:2) = [];
train = normr(train);


%Finds the nubmer of strokes for each user
userLength = zeros(3,N); %stores the number of samples, starting index, and ending index for each user
for i = 1:N
    userIndex = find(test(:,1) == i);      % Finds the indices of every row for a certain user
    %Finds the minimum and maximum of indices length
    minimum = min(userIndex);
    maximum = max(userIndex);
    
    userLength(1,i) = maximum - minimum + 1;   
    userLength(2,i) = minimum;
    userLength(3,i) = maximum;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The following code finds the average vectors for users in the set

matrixSize = size(test);    %rows by columns

%Create a blank matrix that will hold the average vector for a certain user 
averageSet = zeros(N, matrixSize(2) );	


for i = 1:N
	userVectors = test(userLength(2,i):userLength(3,i), :);
	userVectors = sum(userVectors);
	userVectors = userVectors./userLength(1,i);
	averageSet(i,:) = userVectors;
end

averageSet(:,2) = [];
train(:,2) = [];

for i = 1:length(averageSet(:,1))
	averageSet(i,1) = i;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


accuracyVector = zeros(1,41);
accuracy = 0;
minIndex = 0;

M = length(train(:,1));
for i = 1:N
	normMin = 10^9;
	x = averageSet(i,:);
	for j = 1: M
		value = norm(x - train(j,:));
		if (value < normMin)
			normMin = value;
			minIndex = j;
		end
	
	end
	

	accuracyVector(i) = train(minIndex,1);
	if (i == accuracyVector(i))
		accuracy = accuracy + 1;
end









l = 9;	


end