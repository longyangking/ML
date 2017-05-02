function distance = Cosine(p,q)
	distance = p'*q/norm(p)/norm(q);
end