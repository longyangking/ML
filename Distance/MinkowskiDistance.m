function distance = MinkowskiDistance(x1,x2,p)
	distance = (sum((abs(x1-x2))^p))^(1/p);
end