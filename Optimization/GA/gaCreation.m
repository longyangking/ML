function populations = gaCreation(gaopt)
    featuresize = gaopt.Featuresize;
    populationsize = gaopt.PopulationSize;

    LB = gaopt.LB; UB = gaopt.UB; IntCon = gaopt.IntCon;
    LBv = repmat(LB,populationsize,1); UBv = repmat(UB,populationsize,1);
    populations = LBv + rand(populationsize,featuresize).*(UBv - LBv);

    if ~isempty(IntCon)
        intconpops = floor(populations(:,IntCon));
        intconpops = intconpops + 1*(rand(populationsize,length(IntCon))>0.5);

        LBbase = LBv(:,IntCon); UBbase = UBv(:,IntCon);

        posUB = find(intconpops > UBbase);
        intconpops(posUB) = UBbase(posUB);

        posLB = find(intconpops < LBbase);
        intconpops(posLB) = LBbase(posLB);

        populations(:,IntCon) = intconpops;
    end
end