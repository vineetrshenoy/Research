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
test = normr(test);

trainLabels = train(:,1:2);
train(:,1:2) = [];
train = normr(train);



M = length(train(:,1));			%Number of vectors in the training set
N = length(test(:,1));			%Number of vectors in the testing set
accuracyVector = zeros(1,N);
accuracy = 0;
for i = 1:N   			%for every vector in the testing set
	x = test(i,:);
	allNorms = zeros(M,2);		%create a vector that stores all norms for a certain test vector
	for j = 1:M
		allNorms(j,1) = j;
		allNorms(j,2) = norm(x- train(j,:));
	end

	[normMode, frequency] = NnearNeighbors(allNorms,149,trainLabels);
	accuracyVector(i) = normMode;

	if (accuracyVector(i) == testLabels(i,1))
		accuracy = accuracy + 1;
	end

end