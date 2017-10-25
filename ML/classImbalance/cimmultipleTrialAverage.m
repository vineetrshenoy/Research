%This function runs the treeStrokeDistribution function multiple times and plots the results

%INPUT: The full matrix, 
%OUTPUT: graph of the distribution

function [super, avg] = cimmultipleTrialAverage(fullMatrix, numTrials, numUsers, classifier_type)
	rng(5);
	
	N = numUsers;

	[testSet,train, minimum] = imbalance_split(fullMatrix,41);
	
	
	super = zeros(numTrials,minimum);


	userLength = zeros(3,N); %stores the number of samples, starting index, and ending index for each user

	for i = 1:numTrials

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		[testSet,trainSet, minimum] = imbalance_split(fullMatrix,41);

		for i = 1:N
		    userIndex = find(testSet(:,1) == i);      % Finds the indices of every row for a certain user
		    %Finds the minimum and maximum of indices length
		    minimum = min(userIndex);
		    maximum = max(userIndex);
		    
		    userLength(1,i) = maximum - minimum + 1;   
		    userLength(2,i) = minimum;
		    userLength(3,i) = maximum;
		end



		testLabels = testSet(:,1);
		testSet(:,1:2) = [];
		testSet = normr(testSet);


		trainLabels = trainSet(:,1);
		trainSet(:,1:2) = [];
		trainSet = normr(trainSet);



		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		accuracy_vec = 0;
		switch classifier_type
			case 'classification_tree'
				tree = fitctree(trainSet, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				accuracy_vec = cimtreeStrokeDistribution(tree, userLength, testSet, testLabels);
			case 'lda_classifier'
				classifier_lda = fitcdiscr(trainSet, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				accuracy_vec = cimldaStrokeDistribution(classifier_lda, userLength, testSet, testLabels);
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
				[accuracy, one_count] = svmStrokeDistribution(svm_classifier, numUsers, i, testSet, testLabels);
		end
		
		



		super(i,:) = accuracy_vec;


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