%This function runs the treeStrokeDistribution function multiple times and plots the results

%INPUT: The full matrix, 
%OUTPUT: graph of the distribution

function [super, avg] = multipleTrialAverage(fullMatrix, numTrials, numUsers, classifier_type)
	rng(5);
	N = numUsers;

	[testSet,train] = test_train_split(fullMatrix,41);
	
	M = length(testSet(:,1))/N; %Number of test vectors per user
	R = length(train(:,1))/N; %Number of train vectors per user
	super = zeros(numTrials,M);



	for i = 1:numTrials

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		[testSet,train] = test_train_split(fullMatrix,41);

		testLabels = testSet(:,1);
		testSet(:,1:2) = [];
		testSet = normr(testSet);


		trainLabels = train(:,1);
		train(:,1:2) = [];
		train = normr(train);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		accuracy_vec = 0;
		switch classifier_type
			case 'classification_tree'
				tree = fitctree(train, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				accuracy_vec = treeStrokeDistribution(tree, numUsers, testSet, testLabels);
				super(i,:) = accuracy_vec;
			case 'lda_classifier'
				classifier_lda = fitcdiscr(train, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				accuracy_vec = ldaStrokeDistribution(classifier_lda, numUsers, testSet, testLabels);
				super(i,:) = accuracy_vec;
			case 'svm_classifier'
				testLabels(:) = 0;
				trainLabels(:) = 0;

				train_start =  (i - 1) * R + 1;  %finding the current user start point (train)
				train_end = train_start + (R - 1);   %find the current user end point (train)
				trainLabels(train_start:train_end) = 1;		%Setting the new label

				test_start =  (i - 1) * M + 1; %finding the current user start point (test)
				test_end = test_start + (M - 1); %finding the current user end point (test)
				testLabels(test_start:test_end) = 1;		%Setting the new label


				svm_classifier = fitcsvm(train, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				[accuracy] = svmStrokeDistribution(svm_classifier, numUsers, i, testSet, testLabels);
				super(i, 1) = accuracy
		end
		
		



		


	end

	avg = mean(super);

	%{
	figure(1);
	x = 1:M;

	hold on;
	title({'Cross-Validated (K = 20) CART decision tree'});
	xlabel({'Number of samples before classification'});
	ylabel('Classification percentage');
	xlim([0 (M + 3)])
	ylim([0 1.1])
	plot(x, avg, 'r.', 'MarkerSize', 10);
	hold off;
	%}

end