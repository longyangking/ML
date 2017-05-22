function parents = gaSelection(parentsize,ga,gaopt)
    if ~ga.isMultiobjectives
        % Single-Objective Selection
        scaledfitness = gaopt.Fitnessscalefunction(ga.Fitness);
        parents = zeros(1,parentsize);
        for i = 1:parentsize
            individuals = randperm(gaopt.PopulationSize,gaopt.TournamentSize);
            [value,index] = max(scaledfitness(individuals));
            parents(i) = individuals(index);
        end
    else
        % Multiple-Objectives Selection
        [values,indexs] = sortrows([ga.Ranks,-ga.Distances]);

        parents = zeros(parentsize,1);
        for i = 1:parentsize
            individuals = randperm(gaopt.PopulationSize,gaopt.TournamentSize);
            [value,index] = min(indexs(individuals));
            parents(i) = individuals(index);
        end
    end
end
