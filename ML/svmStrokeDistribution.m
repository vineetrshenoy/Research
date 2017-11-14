%This function takes a classifier, testSet and testLabels and creates an accuracy vectors
%based on the number of strokes before classification

%INPUT: classifier, testSet, testLabels
%OUTPUT: accuracy vector

function [accuracy] = svmStrokeDistribution(classifier_svm,numUsers, user_id, testSet, testLabels)


N = numUsers;
M = length(testLabels)/N; %Number of test vectors per user




A = length(testLabels);

%% Simple accuracy test -- total correct out of total number

vec = predict(classifier_svm.Trained{1}, testSet(1:end,:));

super = zeros(A, 1);
%%accuracy test via multiple strokes

predict_start =  (user_id - 1) * M + 1; %The index at which to start prediction
predict_end = predict_start + (M - 1);

super(predict_start:predict_end, 1) = 1;


accuracy = (vec == super);

accuracy = sum(accuracy);
accuracy = accuracy/A;



end