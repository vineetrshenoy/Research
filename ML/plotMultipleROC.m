	function [auc_vec] = plotMultipleROC(trainExtended, trainPolar, trainCart, scoreExtended, scorePolar, scoreCart, trialNum, cType)

	mkdir(sprintf('%sROC', cType));
	auc_vec = zeros(3,41);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	scoreExtendedMat = scoreExtended(:, [2:end]);
	diffExtendedScore = scoreExtended(:, 1) - max(scoreExtendedMat, [], 2);

	scorePolarMat = scorePolar(:, [2:end]);
	diffPolarScore = scorePolar(:, 1) - max(scorePolarMat, [], 2);

	scoreCartMat = scoreCart(:, [2:end]);
	diffCartScore = scoreCart(:, 1) - max(scoreCartMat, [], 2);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[Xex,Yex,Tex,AUCex,OPTROCPTex] = perfcurve(trainExtended,diffExtendedScore, 1);
	[Xp,Yp,Tp,AUCp,OPTROCPTp] = perfcurve(trainPolar,diffPolarScore, 1);
	[Xc,Yc,Tc,AUCc,OPTROCPTc] = perfcurve(trainCart,diffCartScore, 1);


	figure('name', sprintf('%s, trial %d User 1', cType, trialNum), 'visible', 'off');
	hold on
	
	plot(Xex,Yex, 'MarkerSize', 12)
	plot(Xp,Yp, 'MarkerSize', 12)
	plot(Xc,Yc, 'MarkerSize', 12)

	%plot(OPTROCPTex(1),OPTROCPTex(2),'ro')
	%plot(OPTROCPTp(1),OPTROCPTp(2),'ro')
	%plot(OPTROCPTc(1),OPTROCPTc(2),'ro')

	xlabel('False positive rate')
	ylabel('True positive rate')
	title(sprintf('ROC %s, trial %d, user 1', cType, trialNum));
	legend(['Extended Features -- AUC=' num2str(AUCex)], ['Polar Features -- AUC=' num2str(AUCp)], ['Cartesian Features -- AUC=' num2str(AUCc)], 'Location','southeast');
	saveas(gcf,sprintf('%sROC/ROC_%s_trial_%d,_user_1.png', cType, cType, trialNum))
	hold off
	
	auc_vec(1,1) = AUCex;
	auc_vec(2,1) = AUCp;
	auc_vec(3,1) = AUCc;

	for k = 2:40
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		scoreExtendedMat = scoreExtended(:, [1:(k-1), (k+1):end]);
		diffExtendedScore = scoreExtended(:, k) - max(scoreExtendedMat, [], 2);

		scorePolarMat = scorePolar(:, [1:(k-1), (k+1):end]);
		diffPolarScore = scorePolar(:, k) - max(scorePolarMat, [], 2);

		scoreCartMat = scoreCart(:, [1:(k-1), (k+1):end]);
		diffCartScore = scoreCart(:, k) - max(scoreCartMat, [], 2);
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		[Xex,Yex,Tex,AUCex,OPTROCPTex] = perfcurve(trainExtended,diffExtendedScore, k);
		[Xp,Yp,Tp,AUCp,OPTROCPTp] = perfcurve(trainPolar,diffPolarScore, k);
		[Xc,Yc,Tc,AUCc,OPTROCPTc] = perfcurve(trainCart,diffCartScore, k);
		figure('name', sprintf('%s, trial %d User %d', cType, trialNum, k), 'visible', 'off');
		hold on;
		plot(Xex,Yex, 'MarkerSize', 12)
		plot(Xp,Yp, 'MarkerSize', 12)
		plot(Xc,Yc, 'MarkerSize', 12)

		%plot(OPTROCPTex(1),OPTROCPTex(2),'ro')
		%plot(OPTROCPTp(1),OPTROCPTp(2),'ro')
		%plot(OPTROCPTc(1),OPTROCPTc(2),'ro')

		xlabel('False positive rate')
		ylabel('True positive rate')
		title(sprintf('ROC %s, trial %d, user %d',cType, trialNum, k));
		legend(['Extended Features -- AUC=' num2str(AUCex)], ['Polar Features -- AUC=' num2str(AUCp)], ['Cartesian Features -- AUC=' num2str(AUCc)], 'Location','southeast');
		saveas(gcf, sprintf('%sROC/ROC_%s_trial_%d,_user_%d.png',cType , cType, trialNum, k))
		hold off
		
		auc_vec(1,k) = AUCex;
		auc_vec(2,k) = AUCp;
		auc_vec(3,k) = AUCc;


	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	scoreExtendedMat = scoreExtended(1:40);
	diffExtendedScore = scoreExtended(:, 41) - max(scoreExtendedMat, [], 2);

	scorePolarMat = scorePolar(1:40);
	diffPolarScore = scorePolar(:, 41) - max(scorePolarMat, [], 2);

	scoreCartMat = scoreCart(1:40);
	diffCartScore = scoreCart(:, 41) - max(scoreCartMat, [], 2);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[Xex,Yex,Tex,AUCex,OPTROCPTex] = perfcurve(trainExtended,diffExtendedScore, 41);
	[Xp,Yp,Tp,AUCp,OPTROCPTp] = perfcurve(trainPolar,diffPolarScore, 41);
	[Xc,Yc,Tc,AUCc,OPTROCPTc] = perfcurve(trainCart,diffCartScore, 41);
	figure('name', sprintf('%s, trial %d User 41', cType, trialNum), 'visible', 'off');
	hold on;
	plot(Xex,Yex, 'MarkerSize', 12)
	plot(Xp,Yp, 'MarkerSize', 12)
	plot(Xc,Yc, 'MarkerSize', 12)

	%plot(OPTROCPTex(1),OPTROCPTex(2),'ro')
	%plot(OPTROCPTp(1),OPTROCPTp(2),'ro')
	%plot(OPTROCPTc(1),OPTROCPTc(2),'ro')

	xlabel('False positive rate')
	ylabel('True positive rate')
	title(sprintf('ROC %s, trial %d, user 41', cType,  trialNum));
	legend(['Extended Features -- AUC=' num2str(AUCex)], ['Polar Features -- AUC=' num2str(AUCp)], ['Cartesian Features -- AUC=' num2str(AUCc)], 'Location','southeast');
	saveas(gcf,sprintf('%sROC/ROC_%s_trial_%d,_user_41.png', cType, cType, trialNum))
	hold off
	
	auc_vec(1,41) = AUCex;
	auc_vec(2,41) = AUCp;
	auc_vec(3,41) = AUCc;
	auc_vec = mean(auc_vec,2);
	x = 4;


end
