function coefficient = TanimotoCoefficient(a,b)
	coefficient = 1 - (a'*b)/(sqrt(sum(a.^2))+sqrt(sum(b.^2))-a'*b);
end