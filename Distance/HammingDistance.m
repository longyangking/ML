function distance = HammingDistance(s1,s2)
	[M1,N1] = size(s1); [M2,N2] = size(s2); N = N1;
	distance = 0;
	if N1 ~= N2
		return
	end
	for i = 1:N
		if strcmp(s1(i),s2(i))
			distance = distance + 1;
		end
	end
end