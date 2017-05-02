function coefficient = JaccardSimilarityCoefficient(A,B)
	[M1,N1] = size(intersect(A,B));
	[M2,N2] = size(union(A,B));
	coefficient = (N1*M1)/(N2*M2);
end