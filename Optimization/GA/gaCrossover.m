function populations = gaCrossover(ga,gaopt)
    if ~ga.isMultiobjectives
        % Single-objective
        num = floor((gaopt.PopulationSize - gaopt.EliteCount)*gaopt.CrossRatio);
        indexs = gaopt.Selectfunction(num,ga,gaopt);
        parents = ga.Populations(indexs,:);
        populations = zeros(num,gaopt.Featuresize);

        popsize = 1;
        while popsize <= num
            fatherindex = randi(num);
            motherindex = randi(num);
            while fatherindex == motherindex
                motherindex = randi(num);
            end

            father = parents(fatherindex,:);
            mother = parents(motherindex,:);

            son = father + (mother - father).*rand(1,gaopt.Featuresize);
            girl = mother + (father - mother).*rand(1,gaopt.Featuresize);
            
            populations(popsize,:) = son;
            if popsize + 1 <= num
                populations(popsize+1,:) = girl;
            end
            popsize = popsize + 2;
        end

        % LB, UB, Integer Restriction
        LB = gaopt.LB; UB = gaopt.UB; IntCon = gaopt.IntCon;
        LBbase = repmat(LB,num,1); UBbase = repmat(UB,num,1);

        if ~isempty(IntCon)
            intconpops = floor(populations(:,IntCon));
            intconpops = intconpops + 1*(rand(num,length(IntCon))>0.5);
            populations(:,IntCon) = intconpops;
        end

        posUB = find(populations > UBbase);
        populations(posUB) = UBbase(posUB);

        posLB = find(populations < LBbase);
        populations(posLB) = LBbase(posLB);
    else
        % Multiple-objective
        num = gaopt.PopulationSize;
        indexs = gaopt.Selectfunction(num,ga,gaopt);
        parents = ga.Populations(indexs,:);
        populations = zeros(num,gaopt.Featuresize);

        popsize = 1;
        while popsize <= num
            fatherindex = randi(num);
            motherindex = randi(num);
            while fatherindex == motherindex
                motherindex = randi(num);
            end

            father = parents(fatherindex,:);
            mother = parents(motherindex,:);

            son = father + (mother - father).*rand(1,gaopt.Featuresize);
            girl = mother + (father - mother).*rand(1,gaopt.Featuresize);

            populations(popsize,:) = son;
            if popsize + 1 <= num
                populations(popsize+1,:) = girl;
            end
            popsize = popsize + 2;
        end

        % LB, UB, Integer Restriction
        LB = gaopt.LB; UB = gaopt.UB; IntCon = gaopt.IntCon;
        LBbase = repmat(LB,num,1); UBbase = repmat(UB,num,1);

        if ~isempty(IntCon)
            intconpops = floor(populations(:,IntCon));
            intconpops = intconpops + 1*(rand(num,length(IntCon))>0.5);
            populations(:,IntCon) = intconpops;
        end

        posUB = find(populations > UBbase);
        populations(posUB) = UBbase(posUB);

        posLB = find(populations < LBbase);
        populations(posLB) = LBbase(posLB);
    end
end

