function [auc_vec] = plotROC(classifier, trainLabels, score, trialNum)

	mkdir('treeROC');
	auc_vec = zeros(1,41);


	scoreMat = score(:, [2:end]);
	diffscore = score(:, 1) - max(scoreMat, [], 2);

	[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,diffscore, 1);
	figure('name', sprintf('Classification Tree, trial %d User 1', trialNum),'visible', 'off');
	plot(X,Y)
	hold on
	plot(OPTROCPT(1),OPTROCPT(2),'ro')
	xlabel('False positive rate')
	ylabel('True positive rate')
	title(sprintf('ROC Curve Classification Trees, trial %d, user 1', trialNum));
	legend(['AUC=' num2str(AUC)], 'Operating Point');
	saveas(gcf,sprintf('treeROC/ROC_Trees_trial_%d,_user_1.png', trialNum))
	hold off

	auc_vec(1) = AUC;

	for k = 2:40

		scoreMat = score(:, [1:(k-1), (k+1):end]);
		diffscore = score(:, k) - max(scoreMat, [], 2);


		[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,diffscore, k);
		figure('name', sprintf('Classification Tree, trial %d User %d', trialNum, k), 'visible', 'off');
		plot(X,Y)
		hold on
		plot(OPTROCPT(1),OPTROCPT(2),'ro')
		xlabel('False positive rate')
		ylabel('True positive rate')
		title(sprintf('ROC Curve Classification Trees, trial %d, user %d', trialNum, k));
		legend(['AUC=' num2str(AUC)], 'Operating Point');
		saveas(gcf, sprintf('treeROC/ROC_Trees,_trial_%d,_user_%d.png', trialNum, k))
		hold off

		auc_vec(k) = AUC;


	end


	scoreMat = score(1:40);
	diffscore = score(:, 41) - max(scoreMat, [], 2);

	[X,Y,T,AUC,OPTROCPT] = perfcurve(trainLabels,diffscore, 41);
	figure('name', sprintf('Classification Tree, trial %d User 41', trialNum));
	plot(X,Y)
	hold on
	plot(OPTROCPT(1),OPTROCPT(2),'ro')
	xlabel('False positive rate')
	ylabel('True positive rate')
	title(sprintf('ROC Curve Classification Trees, trial %d, user 41', trialNum));
	legend(['AUC=' num2str(AUC)], 'Operating Point');
	saveas(gcf,sprintf('treeROC/ROC_Trees_trial_%d,_user_41.png', trialNum))
	hold off

	auc_vec(41) = AUC;



end
