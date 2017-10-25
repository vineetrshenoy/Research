%This function takes a classifier, testSet and testLabels and creates an accuracy vectors
%based on the number of strokes before classification

%INPUT: classifier, testSet, testLabels
%OUTPUT: accuracy vector

function [accuracy_vec] = cimtreeStrokeDistribution(classifier,userLength, testSet, testLabels)

M = 37;
N = 41;

super = zeros(N,M);


%This forloop creates a matrix of labels per stroke and user (number of user x number of strokes)
%the (i,j) position stores the most common label for that many strokes
% i.e spot (12,20) stores the most frequent label for user 12 after analyzing 12 strokes
for j = 1:N  %From user 1 to user N

	%vector_label = zeros(1,M);	%Vector storing label for each test vector
	predict_start =  userLength(2,j); %The index at which to start prediction
	predict_end = predict_start + (M - 1);
	
	
	% Creates the vector of labels for the first 37 strokes
	vector_label = predict(classifier.Trained{1}, testSet(predict_start:predict_end,:));
	
	for k = 1:M
		temp_vec = vector_label(1:k); 	%takes a subset of the vector
		most = mode(temp_vec);	% find the mode
		super(j,k) = most;	%records the mode

	end


end


accuracy_vec = zeros(1,M);	%stores the accuracy for each user based on the number of strokes
user_labels = 1:N;
user_labels = transpose(user_labels);

for k = 1:M
	
	col_data = super(:,k);
	x = (user_labels == col_data);
	accuracy = sum(x);


	accuracy_vec(k) = accuracy/N;

end





end