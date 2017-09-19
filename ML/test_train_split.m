function [ testSet, trainingSet ] = test_train_split(dataMatrix,N)
%{Creates a test set and training set for the data
%INPUT: 
    %dataMatrix: the features extracted earlier
    %N, the number of users
%OUTPUT:   
    %testSet: Random subset of dataMatrix (20%)
    %trainingSet: Random subset of dataMatrix(80%)

%}

userLength = zeros(3,N); %stores the number of samples, starting index, and ending index for each user
for i = 1:N
    userIndex = find(dataMatrix(:,1) == i);      % Finds the indices of every row for a certain user
    %Finds the minimum and maximum of indices length
    minimum = min(userIndex);
    maximum = max(userIndex);
    
    userLength(1,i) = maximum - minimum + 1;   
    userLength(2,i) = minimum;
    userLength(3,i) = maximum;
end



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




end

