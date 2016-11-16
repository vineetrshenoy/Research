function [accuracy] = simpleNorm(fullMatrix)
%This function takes a full data matrix, creates a 
%testing and training set, and computes a simplie norm error,
%It averages the strokes for a user in the test set and computes
%the norm against every vector in the training set
N = 41;

%Obtains our test and training set
[test,train] = test_train_split(fullMatrix,41);



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


% The following code finds the average vectors for users in the set

matrixSize = size(test);
%Create a blank matrix that will hold the average vector for a certain user 
averageSet = zeros(N, matrixSize(2) );	


v = zeros(1,matrixSize(2));
averageLength = length(userLength);	
N = length(averageSet(:,1));
for i = 1:N
	
	v = zeros(1,matrixSize(2));	

	%Sum all the vectors for a certain user
	for j = userLength(2,i) : userLength(3,i)

		v = v + test(j,:);

	end

	a = userLength(1,i);
	v = v ./ userLength(1,i); %divide by the numbers of vectors
	averageSet(i,:) = v;		%Assign to location i in th average set
	
end

for i = 1:41
	averageSet(i,1) = i;	
end


%Performing the norm calculation
accuracyVector = zeros(1, 41);
accuracy = 0;
minIndex = 0;
M = length(averageSet(:,1));  %Length of the set of average vectors
N = length(train(:,1));		%Length of the training set
count = 0;

%For every vector in the average test set
for i = 1:M

	minIndex = 1;
	minNorm = 10^9;
	x = averageSet(i,:);

	for j = 1:N
		y = train(j,:);
		normValue =  norm(x-y);
		if (normValue < minNorm)
			minNorm = normValue;
			minIndex = j;
		end


	end

	accuracyVector(i) = fullMatrix(minIndex,1);
	if (i == accuracyVector(i))
		accuracy = accuracy + 1;
	end




end

l = 9;	


end