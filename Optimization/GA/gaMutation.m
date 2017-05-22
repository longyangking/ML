function populations = gaMutation(ga,gaopt)
    if ~ga.isMultiobjectives
        % Single-objective
        num = gaopt.PopulationSize - gaopt.EliteCount - floor((gaopt.PopulationSize - gaopt.EliteCount)*gaopt.CrossRatio);
        indexs = gaopt.Selectfunction(num,ga,gaopt);
        populations = ga.Populations(indexs,:);

        populationsize = num; featuresize = gaopt.Featuresize;
        LB = gaopt.LB; UB = gaopt.UB; IntCon = gaopt.IntCon;
        LBv = repmat(LB,populationsize,1); UBv = repmat(UB,populationsize,1);

        pos = rand(populationsize,featuresize) < gaopt.MutationRate;
        randvalues = LBv + rand(populationsize,featuresize).*(UBv - LBv);
        populations(pos) = randvalues(pos);

        % Integer Restriction
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
    else
        % Multiple-objectives
        indexs = gaopt.Selectfunction(gaopt.PopulationSize,ga,gaopt);
        populations = ga.Populations(indexs,:);

        populationsize = gaopt.PopulationSize; featuresize = gaopt.Featuresize;
        LB = gaopt.LB; UB = gaopt.UB; IntCon = gaopt.IntCon;
        LBv = repmat(LB,populationsize,1); UBv = repmat(UB,populationsize,1);

        pos = rand(populationsize,featuresize) < gaopt.MutationRate;
        randvalues = LBv + rand(populationsize,featuresize).*(UBv - LBv);
        populations(pos) = randvalues(pos);

        % Integer Restriction
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
end