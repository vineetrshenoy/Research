<<<<<<< Updated upstream
function [accuracy,test,train] = simpleNorm(fullMatrix)
=======
function [user_mat] = simpleNorm(fullMatrix)
>>>>>>> Stashed changes
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

% inf = isinf(mat);
% t = find(inf(:,:) == 1);
M = length(testLabels)/N; %Number of test vectors per user
test_length = length(testLabels(:,1));
train_length = length(trainLabels(:,1));


norm_mat = zeros(test_length, train_length);

for i = 1: test_length

	x = test(i,:);

	for j = 1: train_length

		norm_mat(i,j) = norm(x - train(j,:));

	end


%Finds the nubmer of strokes for each user
userLength = zeros(3,N); %stores the number of samples, starting index, and ending index for each user
for i = 1:N
    userIndex = find(testLabels(:,1) == i);      % Finds the indices of every row for a certain user
    %Finds the minimum and maximum of indices length
    minimum = min(userIndex);
    maximum = max(userIndex);
    
    userLength(1,i) = maximum - minimum + 1;   
    userLength(2,i) = minimum;
    userLength(3,i) = maximum;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The following code finds the average vectors for users in the set
%{
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
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%norm_mat = norm_mat';
[min_norm, I] = min(norm_mat, [],2);
min_norm = [min_norm,I];


user_mat = zeros(N,M);

for i = 1:M


	for j = 1:N

		vector_label = zeros(1,i);	%Vector storing label for each test vector
		predict_start =  (j - 1) * M + 1; %The index at which to start prediction
		predict_end = predict_start + (i - 1);
		count = 1;	%Counter to store labels in right place
		
		input_vec = min_norm(predict_start:predict_end, :);
		[label, freq] = NnearNeighbors(input_vec, trainLabels);
		user_mat(N,M) = label;

		

	end



end



x = 5;








%{
M = length(train(:,1));			%Number of vectors in the training set
N = length(test(:,1));			%Number of vectors in the testing set
<<<<<<< Updated upstream

accuracyVector = zeros(N,1);
accuracy = 0;
minIndex = 0;


for i = 1:N
	normMin = 10^9;
	x = test(i,:);
	for j = 1: M
		value = norm(x - train(j,:));
		if (value < normMin)
			normMin = value;
			minIndex = j;
		end
	
	end
	

	accuracyVector(i) = trainLabels(minIndex,1);
	if (testLabels(i,1) == accuracyVector(i))
		accuracy = accuracy + 1;
end









l = 9;	
=======
Z = N / X;
accuracyVector = zeros(Z,N);
accuracy = 0;

	
	for j = 1:N   			%for every vector in the testing set
		x = test(j,:);
		allNorms = zeros(M,2);		%create a vector that stores norms between the test vector and every vector in the training set
		for k = 1:M
			allNorms(k,1) = k;
			allNorms(k,2) = norm(x- train(k,:));	%stores the label as well as norm
		end

		for k = 1:
		[normMode, frequency] = NnearNeighbors(allNorms,3,trainLabels);
		accuracyVector(j) = normMode;

		if (accuracyVector(j) == testLabels(j,1))
			accuracy = accuracy + 1;
		end

	end
%}

end
>>>>>>> Stashed changes


