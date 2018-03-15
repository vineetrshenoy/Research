%This function runs the treeStrokeDistribution function multiple times and plots the results

%INPUT: The full matrix, 
%OUTPUT: graph of the distribution

function [super_acc, avg] = multipleTrialAverage(fullMatrix, numTrials, numUsers, classifier_type)
	rng(5);
	N = numUsers;

	[testSet,trainSet] = test_train_split(fullMatrix,41);
	
	M = length(testSet(:,1))/N; %Number of test vectors per user
	R = length(trainSet(:,1))/N; %Number of train vectors per user
	super_acc = zeros(numTrials,M);
	auc_super = zeros(numTrials, 41);



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
			case 'tree'
				tree = fitctree(trainSet, trainLabels);
				[~,score] = resubPredict(tree);
				auc_vec = plotROC(tree, trainLabels, score, i, 'ClassificationTree');
				auc_super(i, :) = auc_vec;
				
				accuracy_vec = treeStrokeDistribution(tree, numUsers, testSet, testLabels);
				super_acc(i,:) = accuracy_vec;


			case 'lda'
				classifier_lda = fitcdiscr(trainSet, trainLabels);
				[~,score] = resubPredict(classifier_lda);
				auc_vec = plotROC(classifier_lda, trainLabels, score, i, 'LDAclassification');
				auc_super(i, :) = auc_vec;

				accuracy_vec = ldaStrokeDistribution(classifier_lda, numUsers, testSet, testLabels);
				super_acc(i,:) = accuracy_vec;
			case 'svm'
				cType = 'SVM';
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

				[X,Y,T,AUC, OPTROCPT] = perfcurve(trainLabels,score(:,2),1);
				%figure('name', sprintf('%s, trial %d User %d', cType, i, user), 'visible', 'off');
				figure(1)
				plot(X,Y)
				hold on
				plot(OPTROCPT(1),OPTROCPT(2),'ro')
				xlabel('False positive rate')
				ylabel('True positive rate')
				title(sprintf('ROC %s, trial %d, user %d',cType, i, user));
				legend(['AUC=' num2str(AUC)], 'Operating Point');
				%saveas(gcf, sprintf('%sROC/ROC_%s_trial_%d,_user_%d.png',cType , cType, trialNum, user))
				hold off

				%classLoss = kfoldLoss(svm_classifier);
				%{
				[accuracy, far, frr] = svmStrokeDistribution2(svm_classifier, numUsers, 1, testSet, testLabels);
				super_acc(i, 1) = accuracy;
				super_acc(i, 2) = far;
				super_acc(i, 3) = frr;
				%}	

			case 'knn'

				knn_classifer = fitcknn(trainSet, trainLabels, 'NumNeighbors', 20, 'Standardize', 1);
				
				[~,score] = resubPredict(knn_classifer);
				auc_vec = plotROC(knn_classifer, trainLabels, score, i, 'kNNclassification');
				auc_super(i, :) = auc_vec;


			case 'nb'

				nb_classifier = fitcnb(trainSet, trainLabels);
				[~,score] = resubPredict(nb_classifier);
				auc_vec = plotROC(nb_classifier, trainLabels, score, i, 'NaiveBayes');
				auc_super(i, :) = auc_vec;


			case 'svm2'

				userTest_indices = find(testLabels(:,1) == i);
				userTrain_indices = find(trainLabels(:,1) == i);

				testLabels(:) = 0;
				trainLabels(:) = 0;

				testLabels(userTest_indices) = 1;
				trainLabels(userTrain_indices) = 1;

				svm_classifier = fitcsvm(trainSet, trainLabels, 'Standardize',true, 'KernelFunction', 'rbf', 'KernelScale','auto');
				svm_classifier = fitPosterior(svm_classifier);
				[~,score] =  resubPredict(svm_classifier);
				%[label, score] = predict(svm_classifier, testSet(1:end,:));

				[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,score(:,2),1);
				figure(i)
				plot(X,Y)
				hold on
				plot(OPTROCPT(1),OPTROCPT(2),'ro')
				xlabel('False positive rate')
				ylabel('True positive rate')
				legend(['AUC=' num2str(AUC)], 'Operating Point');

				%classLoss = kfoldLoss(svm_classifier);
				%[accuracy, far, frr] = svmStrokeDistribution(svm_classifier, numUsers, 1, testSet, testLabels);
				%super_acc(i, 1) = accuracy;
				%super_acc(i, 2) = far;
				%super_acc(i, 3) = frr;




		end
		

	end

	avg = mean(super_acc);


end