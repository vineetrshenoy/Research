%This function takes a classifier, testSet and testLabels and creates an accuracy vectors
%based on the number of strokes before classification

%INPUT: classifier, testSet, testLabels
%OUTPUT: accuracy vector

function [accuracy, one_count] = svmStrokeDistribution(classifier_svm,numUsers, user_id, testSet, testLabels)


N = numUsers;
M = length(testLabels)/N; %Number of test vectors per user




A = length(testLabels);
l = zeros(1, A);
%% Simple accuracy test -- total correct out of total number
accuracy = 0;

vec = predict(classifier_svm.Trained{1}, testSet(1:end,:));




%%accuracy test via multiple strokes
super = zeros(1,M);
vector_label = zeros(1,M);	%Vector storing label for each test vector
predict_start =  (user_id - 1) * M + 1; %The index at which to start prediction
predict_end = predict_start + (M - 1);


x = predict(classifier_svm.Trained{1}, testSet(predict_start:predict_end,:));
ones_array = ones(1,M)';

vector_label == ones_array




end