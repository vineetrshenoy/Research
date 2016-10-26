function [accurary] = simpleNorm(fullMatrix)
%This function takes a full data matrix, creates a 
%testing and training set, and computes a simplie norm error,
%It averages the strokes for a user in the test set and computes
%the norm against every vector in the training set
N = 41;

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

matrixSize = size(test);
averageSet = zeros(N, matrixSize(2) );


v = zeros(1,matrixSize(2));
averageLength = length(userLength);	
for i = 1:length(averageSet(:,1))
	


	for j = userLength(2,i) : userLength(3,i)

		v = v + test(j,:);

	end
	v = v ./ userLength(1,i);
	averageSet(i,:) = v;
	v = zeros(1,matrixSize(2));

end
a = 7;



end