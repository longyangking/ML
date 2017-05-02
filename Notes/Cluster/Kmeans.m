function idx = Kmeans(X,k)
	maxiter = 1000; tolerance = 1e-1;
	[m,n] = size(X);
	idx = zeros(m,1); 
	core = X(round(rand(1,k)*(m-1)+1),:); iter = 1;
	while (iter<=maxiter)
		newcore = zeros(k,n); count = zeros(k,1);
		for i = 1:m
			bestdis = -1;
			for pos = 1:k
				dis = sum(abs(X(i,:)-core(pos,:)).^2);
				if (bestdis <0)||(bestdis > dis)
					bestdis = dis;
					idx(i) = pos;
				end
			end
			newcore(idx(i),:) = newcore(idx(i),:) + X(i,:);
			count(idx(i)) = count(idx(i)) + 1;
		end
		for pos = 1:k
			newcore(k,:) = newcore(k,:)/count(k);
		end
		%idx.'
		%if sum(sum((newcore-core).^2)) < tolerance
		if length(setdiff(newcore, core)) == 0
			break;
		else
			core = newcore;
			iter = iter + 1;
		end
	end
	if (iter >= maxiter)
		disp('Max iteration happen');
	end
end