function [M,F] = NnearNeighbors(allNorms, trainLabels)

	N = length(allNorms(:,1));
	smallestNorms = sortrows(allNorms,1);		%Sorts the rows of A in ascending order of the first column
	%smallestNorms(leastN + 1:end,:) = [];		%Keeps leastN norms and throws away the rest
	for i = 1:N
		label = trainLabels(smallestNorms(i,2),1);	%Finds label in trainLabels that corresponds to index in smallestNorms
		smallestNorms(i,2) = label;					%Sets label in smallestNorms
	end


	[M,F] = mode(smallestNorms(:,2));		%Finds the mode


end
