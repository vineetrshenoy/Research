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
		M = length(testLabels) / numUsers;


		trainLabels = train(:,1);
		train(:,1:2) = [];
		train = normr(train);
		R = length(trainLabels) / numUsers;

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		accuracy_vec = 0;
		switch classifier_type
			case 'classification_tree'
				tree = fitctree(train, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				[label,score] = resubPredict(tree.Trained{1});
				accuracy_vec = treeStrokeDistribution(tree, numUsers, testSet, testLabels);
				super(i,:) = accuracy_vec;
			case 'lda_classifier'
				classifier_lda = fitcdiscr(train, trainLabels, 'CrossVal', 'on', 'KFold', 15);
				accuracy_vec = ldaStrokeDistribution(classifier_lda, numUsers, testSet, testLabels);
				super(i,:) = accuracy_vec;
			case 'svm_classifier'

				[testSet, trainSet, user] = svmSplit(fullMatrix, 41);
				userTest_indices = find(testSet(:,1) == user);
				userTrain_indices = find(trainSet(:,1) == user);

				testLabels = testSet(:,1);
				testSet(:,1:2) = [];
				testSet = normr(testSet);

				trainLabels = trainSet(:,1);
				trainSet(:,1:2) = [];
				trainSet = normr(trainSet);


				testLabels(:) = 0;
				trainLabels(:) = 0;

				testLabels(userTest_indices) = 1;
				trainLabels(userTrain_indices) = 1;


				svm_classifier = fitcsvm(trainSet, trainLabels, 'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
				svm_classifier = fitPosterior(svm_classifier);
				[~,score] =  resubPredict(svm_classifier);
				%[label, score] = predict(svm_classifier, testSet(1:end,:));

				[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(trainLabels,score(:,2),1);
				plot(Xsvm,Ysvm);
				xlabel('False Positive Rate');
				ylabel('True Positive Rate');


				%classLoss = kfoldLoss(svm_classifier);
				[accuracy, far, frr] = svmStrokeDistribution2(svm_classifier, numUsers, 1, testSet, testLabels);
				super(i, 1) = accuracy;
				super(i, 2) = far;
				super(i, 3) = frr;
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