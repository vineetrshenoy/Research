%This function runs the treeStrokeDistribution function multiple times and plots the results

%INPUT: The full matrix, 
%OUTPUT: graph of the distribution

function [super, avg] = multipleTrialAverage(fullMatrix, numTrials, numUsers, classifier_type)
	rng(5);
	N = numUsers;

	[testSet,trainSet] = test_train_split(fullMatrix,41);
	
	M = length(testSet(:,1))/N; %Number of test vectors per user
	R = length(trainSet(:,1))/N; %Number of train vectors per user
	super = zeros(numTrials,M);



	for i = 1:numTrials

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		[testSet,trainSet] = test_train_split(fullMatrix,41);

		testLabels = testSet(:,1);
		testSet(:,1:2) = [];
		testSet = normr(testSet);
		M = length(testLabels) / numUsers;


		trainLabels = trainSet(:,1);
		trainSet(:,1:2) = [];
		trainSet = normr(trainSet);
		R = length(trainLabels) / numUsers;

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		accuracy_vec = 0;
		switch classifier_type
			case 'classification_tree'
				tree = fitctree(trainSet, trainLabels);
				
				[~,score] = resubPredict(tree);
				scoreMat = score(:, [1:2, 4:end]);
				diffscore = score(:, 3) - max(scoreMat, [], 2);


				[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,diffscore, 3);
				figure(1);
				plot(X,Y)
				hold on
				plot(OPTROCPT(1),OPTROCPT(2),'ro')
				xlabel('False positive rate')
				ylabel('True positive rate')
				title('ROC Curve for Classification by Classification Trees')
				hold off
				accuracy_vec = treeStrokeDistribution(tree, numUsers, testSet, testLabels);
				super(i,:) = accuracy_vec;


			case 'lda_classifier'
				classifier_lda = fitcdiscr(trainSet, trainLabels);
				


				[~,score] = resubPredict(classifier_lda);
				scoreMat = score(:, [1, 3:end]);
				diffscore = score(:, 2) - max(scoreMat, [], 2);


				[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,diffscore, 2);
				figure(1);
				plot(X,Y)
				hold on
				plot(OPTROCPT(1),OPTROCPT(2),'ro')
				xlabel('False positive rate')
				ylabel('True positive rate')
				title('ROC Curve for Classification by LDA')
				hold off

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


				svm_classifier = fitcsvm(trainSet, trainLabels, 'Standardize',true, 'KernelFunction', 'rbf', 'KernelScale','auto');
				svm_classifier = fitPosterior(svm_classifier);
				[~,score] =  resubPredict(svm_classifier);
				%[label, score] = predict(svm_classifier, testSet(1:end,:));

				[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(trainLabels,score(:,2),1);
				figure(i)
				title(sprintf('Extended Features, Trial %d', i))
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