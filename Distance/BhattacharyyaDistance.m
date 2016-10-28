function distance = BhattacharyyaDistance(p,q)
%In statistics, the Bhattacharyya distance measures the similarity 
%of two discrete or continuous probability distributions. It is 
%closely related to the Bhattacharyya coefficient which is a measure 
%of the amount of overlap between two statistical samples or populations.
	sigmap = cov(p); sigmaq = cov(q);
	mup = mean(p); muq = mean(q);
	sigma = (sigmap+sigmaq)/2;
	distance = (mup-muq)*inv(sigma)*transpose(mup-muq)/8 + log(det(sigma)/sqrt(det(sigmap)*det(sigmaq)))/2;
end

