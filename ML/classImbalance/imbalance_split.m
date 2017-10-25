function [ testSet, trainingSet, minimum ] = imbalance_split(dataMatrix,N)
%{Creates a test set and training set for the data
%INPUT: 
    %dataMatrix: the features extracted earlier
    %N, the number of users
%OUTPUT:   
    %testSet: Random subset of dataMatrix (20%)
    %trainingSet: Random subset of dataMatrix(80%)

%}

userLength = zeros(4,N); %stores the number of samples, starting index, and ending index for each user
for i = 1:N
    userIndex = find(dataMatrix(:,1) == i);      % Finds the indices of every row for a certain user
    %Finds the minimum and maximum of indices length
    minimum = min(userIndex);
    maximum = max(userIndex);
    
    userLength(1,i) = maximum - minimum + 1;   
    userLength(2,i) = minimum;
    userLength(3,i) = maximum;
    userLength(4,i) = round(0.2*userLength(1,i));
end
minimum = min(userLength(4,:));
testSum = sum(userLength(4,:));  %gets the sum so that a new matrix can be created
matrixSize = size(dataMatrix);

%creates blank matrices with number of columns equal to number of features
testSet = zeros(testSum, matrixSize(2));         
trainingSet = zeros(matrixSize(1) - testSum, matrixSize(2));

%Know where to start placing values
previous_test_start = 1;
previous_train_start = 1;
for i = 1:N
    totalSamples = userLength(1,i);  %total number of values for a certain user
    userIndex = find(dataMatrix(:,1) == i);         %Finds all rows for a certain user
    %Random indices used for training and testing. The number equls minimum
    

    randomIndices = userIndex(randperm(length(userIndex))); %shuffling the numbers
    testNumber = round(0.2*totalSamples);
    trainNumber = totalSamples - testNumber;



    %Indices for the testing. Takes 20% * minimum
    testIndices = randomIndices(1:testNumber);      
    randomIndices(1:testNumber) = [];        
    trainIndices = randomIndices;                   %indices for training
    
    test_end = previous_test_start + testNumber;
    userTestSet = dataMatrix(testIndices,:);    
    testSet(previous_test_start:test_end-1, 1:matrixSize(2)) = userTestSet;
    
    
    train_end = previous_train_start + trainNumber;
    userTrainSet = dataMatrix(trainIndices,:);
    trainingSet(previous_train_start:train_end-1, 1:matrixSize(2)) = userTrainSet;
    

    previous_test_start = test_end;
    previous_train_start = train_end;

end











%{
minimum = min(userLength(1,:)); %Find the minimum number of samples for a user; reference point
testNumber = round(0.2*minimum);   % 20% for test set
trainNumber = minimum - testNumber; % 80% for training set


matrixSize = size(dataMatrix);

%creates blank matrices with number of columns equal to number of features
testSet = zeros(testNumber * (N), matrixSize(2));         
trainingSet = zeros(trainNumber * (N), matrixSize(2));
%For some reason, user 0 is not displaying properly
for i = 1:N
    userIndex = find(dataMatrix(:,1) == i);         %Finds all rows for a certain user
    %Random indices used for training and testing. The number equls minimum
    randomIndices = randi([userIndex(1) userIndex(end)],testNumber + trainNumber,1);    
    %Indices for the testing. Takes 20% * minimum
    testIndices = randomIndices(1:testNumber);      
    randomIndices(1:testNumber) = [];        
    trainIndices = randomIndices;                   %indices for training
    
    
    userTestSet = dataMatrix(testIndices,:);
    testSet((i-1)*testNumber + 1:(i)*testNumber, 1:matrixSize(2)) = userTestSet;
    
    userTrainSet = dataMatrix(trainIndices,:);
    trainingSet((i-1)*trainNumber + 1:(i)*trainNumber, 1:matrixSize(2)) = userTrainSet;
    

end

allo = 5;

%}


end

