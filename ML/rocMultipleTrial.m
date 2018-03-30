%This function runs the treeStrokeDistribution function multiple times and plots the results

%INPUT: The full matrix, 
%OUTPUT: graph of the distribution

function [super_acc, auc_super] = rocMultipleTrial(extendedFeatures, polarMat, cartMat, numTrials, numUsers, classifier_type)
	rng(5);
	N = numUsers;

	[testSet,trainSet] = test_train_split(extendedFeatures,41);
	
	M = length(testSet(:,1))/N; %Number of test vectors per user
	R = length(trainSet(:,1))/N; %Number of train vectors per user
	super_acc = zeros(numTrials,M);
	auc_super = zeros(numTrials, 3);



	for i = 1:numTrials

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		[testExtended,trainExtended] = test_train_split(extendedFeatures,41);

		testLabExtended = testExtended(:,1);
		testExtended(:,1:2) = [];
		testExtended = normr(testExtended);
		


		trainLabExtended = trainExtended(:,1);
		trainExtended(:,1:2) = [];
		trainExtended = normr(trainExtended);
		
		%--------------------------------------------------------------------------

		[testPolar,trainPolar] = test_train_split(polarMat,41);

		testLabPolar = testPolar(:,1);
		testPolar(:,1:2) = [];
		testPolar = normr(testPolar);
		


		trainLabPolar = trainPolar(:,1);
		trainPolar(:,1:2) = [];
		trainPolar = normr(trainPolar);
		
		%--------------------------------------------------------------------------

		[testCart,trainCart] = test_train_split(cartMat,41);

		testLabCart = testCart(:,1);
		testCart(:,1:2) = [];
		testCart = normr(testCart);
		


		trainLabCart = trainCart(:,1);
		trainCart(:,1:2) = [];
		trainCart = normr(trainCart);
		
		%--------------------------------------------------------------------------





		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		accuracy_vec = 0;
		switch classifier_type
			case 'tree'
				%--------------------------------------------------------
				treeExtended = fitctree(trainExtended, trainLabExtended);
				[~,scoreExtended] = resubPredict(treeExtended);

				treePolar = fitctree(trainPolar, trainLabPolar);
				[~,scorePolar] = resubPredict(treePolar);

				treeCart = fitctree(trainCart, trainLabCart);
				[~,scoreCart] = resubPredict(treeCart);
				%--------------------------------------------------------

				auc_vec = plotMultipleROC(trainLabExtended, trainLabPolar, trainLabCart, scoreExtended, scorePolar, scoreCart, i, 'mClassificationTreePolar');
				auc_super(i, :) = auc_vec;
				
				%accuracy_vec = treeStrokeDistribution(tree, numUsers, testSet, testLabels);
				%super_acc(i,:) = accuracy_vec;


			case 'lda'
				%--------------------------------------------------------
				ldaExtended = fitcdiscr(trainExtended, trainLabExtended);
				[~,scoreExtended] = resubPredict(ldaExtended);

				ldaPolar = fitcdiscr(trainPolar, trainLabPolar);
				[~,scorePolar] = resubPredict(ldaPolar);

				ldaCart = fitcdiscr(trainCart, trainLabCart);
				[~,scoreCart] = resubPredict(ldaCart);
				%--------------------------------------------------------



				auc_vec = plotMultipleROC(trainLabExtended, trainLabPolar, trainLabCart, scoreExtended, scorePolar, scoreCart, i, 'mLDAclassificationPolar');
				auc_super(i, :) = auc_vec;

				%accuracy_vec = ldaStrokeDistribution(classifier_lda, numUsers, testSet, testLabels);
				%super_acc(i,:) = accuracy_vec;
			case 'svm'
				svmROC(extendedFeatures, 41);

				%classLoss = kfoldLoss(svm_classifier);
				%{
				[accuracy, far, frr] = svmStrokeDistribution2(svm_classifier, numUsers, 1, testSet, testLabels);
				super_acc(i, 1) = accuracy;
				super_acc(i, 2) = far;
				super_acc(i, 3) = frr;
				%}	

			case 'knn'



				%--------------------------------------------------------
				knnExtended = fitcknn(trainExtended, trainLabExtended, 'NumNeighbors', 20, 'Standardize', 1);
				[~,scoreExtended] = resubPredict(knnExtended);

				knnPolar = fitcknn(trainPolar, trainLabPolar, 'NumNeighbors', 20, 'Standardize', 1);
				[~,scorePolar] = resubPredict(knnPolar);

				knnCart = fitcknn(trainCart, trainLabCart, 'NumNeighbors', 20, 'Standardize', 1);
				[~,scoreCart] = resubPredict(knnCart);
				%--------------------------------------------------------


				auc_vec = plotMultipleROC(trainLabExtended, trainLabPolar, trainLabCart, scoreExtended, scorePolar, scoreCart, i, 'mkNNclassificationPolar');
				auc_super(i, :) = auc_vec;


			case 'nb'


				%--------------------------------------------------------
				nbExtended = fitcnb(trainExtended, trainLabExtended);
				[~,scoreExtended] = resubPredict(nbExtended);

				nbPolar = fitcnb(trainPolar, trainLabPolar);
				[~,scorePolar] = resubPredict(nbPolar);

				nbCart = fitcnb(trainCart, trainLabCart);
				[~,scoreCart] = resubPredict(nbCart);
				%--------------------------------------------------------


				auc_vec = plotMultipleROC(trainLabExtended, trainLabPolar, trainLabCart, scoreExtended, scorePolar, scoreCart, i, 'NaiveBayesPolar');
				auc_super(i, :) = auc_vec;


			case 'svm2'
				cType = 'SVMPolar';
				labelsTest = testLabels;
				labelsTrain = trainLabels;


				mkdir(sprintf('%sROC', cType))
				for k = 1:41

					testLabels = labelsTest;
					trainLabels = labelsTrain;

					userTest_indices = find(testLabels(:,1) == k);
					userTrain_indices = find(trainLabels(:,1) == k);

					testLabels(:) = 0;
					trainLabels(:) = 0;

					testLabels(userTest_indices) = 1;
					trainLabels(userTrain_indices) = 1;

					svm_classifier = fitcsvm(trainSet, trainLabels, 'Standardize',true, 'KernelFunction', 'rbf', 'KernelScale','auto');
					svm_classifier = fitPosterior(svm_classifier);
					[~,score] =  resubPredict(svm_classifier);
					%[label, score] = predict(svm_classifier, testSet(1:end,:));

					[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,score(:,2),1);
					figure('name', sprintf('%s, trial %d User %d', cType, i, k), 'visible', 'off');
					%figure(1)
					plot(X,Y)
					hold on
					plot(OPTROCPT(1),OPTROCPT(2),'ro')
					xlabel('False positive rate')
					ylabel('True positive rate')
					title(sprintf('ROC %s, trial %d, user %d',cType, i, k));
					legend(['AUC=' num2str(AUC)], 'Operating Point');
					saveas(gcf, sprintf('%sROC/ROC_%s_trial_%d,_user_%d.png',cType , cType, i, k))
					hold off


					auc_super(i,k) = AUC;
				end

				

				

		end
		

	end

	avg = mean(super_acc);


end