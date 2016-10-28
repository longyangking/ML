function coefficient = PearsonCorrelationCoefficient(X,Y)
	coefficient = mean((X-mean(X)).*(Y-mean(Y)))/std(X)/std(Y);
end