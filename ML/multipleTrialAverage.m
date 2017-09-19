%This function runs the treeStrokeDistribution function multiple times and plots the results

%INPUT: The full matrix, 
%OUTPUT: graph of the distribution

function [super, avg] = multipleTrialAverage(fullMatrix, numTrials, numUsers)
	rng(5);
	N = numUsers;

	[testSet,train] = test_train_split(fullMatrix,41);
	
	M = length(testSet(:,1))/N; %Number of test vectors per user

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

		tree = fitctree(train, trainLabels, 'CrossVal', 'on', 'KFold', 20);
		accuracy_vec = treeStrokeDistribution(tree, numUsers, testSet, testLabels);
		super(i,:) = accuracy_vec;


	end

	avg = mean(super);

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

end