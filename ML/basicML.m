function [value ] = basicML(fileName )
%This function provides a basic ML understanding of the data.


%Loads the data from the MAT file and stores as a matrix
data = load(fileName);
features = cell2mat(struct2cell(data));

%creates the test/training set split
[test,train] = test_train_split(features,41);
%[trainInd,valInd,testInd] = dividerand(21174,0.6,0.2,0.2);




value = 0;


end

