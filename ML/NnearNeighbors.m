function [M,F] = NnearNeighbors(allNorms, leastN, trainLabels)

	smallestNorms = sortrows(allNorms,2);		%Sorts the rows of A in ascending order of the first column
	smallestNorms(leastN + 1:end,:) = [];		%Keeps leastN norms and throws away the rest
	for i = 1:leastN
		label = trainLabels(smallestNorms(i,1),1);	%Finds label in trainLabels that corresponds to index in smallestNorms
		smallestNorms(i,2) = label;					%Sets label in smallestNorms
	end


	[M,F] = mode(smallestNorms(:,2));		%Finds the mode


end
