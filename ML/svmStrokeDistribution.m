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

frr = 0;
%Calculating the False Rejection Rate
for i = predict_start:predict_end

	if accuracy(i) == 0
		frr = frr + 1;
	end

end

frr = frr/M;

%Calculating the False Acceptance Rate
far = 0
for j = 1:predict_start - 1

	if accuracy(j) == 0
		far = far + 1;
	end

end

for j = predict_end + 1:length(testSet)

	if accuracy(j) == 0
		far = far + 1;
	end

end
far = far/(A -M);



accuracy = sum(accuracy);
accuracy = accuracy/A;



end