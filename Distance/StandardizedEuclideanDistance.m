function distance = StandardizedEuclideanDistance(x1,x2)
	X1 = (x1-mean(x1))/std(x1);
	X2 = (x2-mean(x2))/std(x2);
	distance = sqrt(sum((X1-X2).^2));
end