clear,clc
load('matricesAndStructures/extendFeat.mat')
load('matricesAndStructures/polarMat.mat')
load('matricesAndStructures/cartMat.mat')



x = 1:M;
%{}
[superOne, avgOne] = multipleTrialAverage(extendedFeatures, 100, 41, 'classification_tree')
figure(1);
hold on;
title({'Cross-Val (K = 15) CART decision tree, Polar + Cartesian, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgOne, 'r.', 'MarkerSize', 10);
hold off;
legend('Cartesian + Polar')


[superTwo, avgTwo] = multipleTrialAverage(polarMat, 100, 41, 'classification_tree')
figure(2);
hold on;
title({'Cross-Val (K = 15) CART decision tree, Polar Only, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgTwo, 'b.', 'MarkerSize', 10);
hold off;
legend('Polar')

[superThree, avgThree] = multipleTrialAverage(cartMat, 100, 41, 'classification_tree')
figure(3);
hold on;
title({'Cross-Val (K = 15) CART decision tree, Cartesian Only, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgThree, 'k.', 'MarkerSize', 10);
hold off;
legend('Cartesian')
%}

figure(4);
hold on;
title({'Cross-Val (K = 15) CART decision tree, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgOne, 'r.');
plot(x, avgTwo, 'b.')
plot(x, avgThree, 'k.')
hold off;
legend('Polar + Cartesian', 'Polar Only', 'Cartesian Only')


[superFive, avgFive] = multipleTrialAverage(extendedFeatures, 100, 41, 'lda_classifier')
figure(5);
hold on;
title({'Cross-Val (K = 15) LDA classifier, Polar + Cartesian, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgFive, 'r.', 'MarkerSize', 10);
hold off;
legend('Cartesian + Polar')


[superSix, avgSix] = multipleTrialAverage(polarMat, 100, 41, 'lda_classifier')
figure(6);
hold on;
title({'Cross-Val (K = 15) LDA classifier, Polar, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgSix, 'r.', 'MarkerSize', 10);
hold off;
legend('Polar')

[superSeven, avgSeven] = multipleTrialAverage(cartMat, 100, 41, 'lda_classifier')
figure(7);
hold on;
title({'Cross-Val (K = 15) LDA classifier, Cartesian, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgSeven, 'r.', 'MarkerSize', 10);
hold off;
legend('Cartesian')



figure(8);
hold on;
title({'Cross-Val (K = 15) LDA classifier, 100 Trials'});
xlabel({'Number of samples before classification'});
ylabel('Classification percentage');
xlim([0 (M + 3)])
ylim([0 1.1])
plot(x, avgFive, 'r.');
plot(x, avgSix, 'b.')
plot(x, avgSeven, 'k.')
hold off;
legend('Polar + Cartesian', 'Polar Only', 'Cartesian Only')