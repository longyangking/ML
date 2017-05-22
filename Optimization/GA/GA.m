function ga = GA(gaopt)
    % Read GA Options Structures
    func = gaopt.Fitnessfunction;
    featuresize = gaopt.Featuresize;
    targetsize = gaopt.Targetsize;
    constraintfunction = gaopt.Constraintfunction;
    populationsize = gaopt.PopulationSize;
    initialpopulations = gaopt.InitialPopulations;
    elitecount = gaopt.EliteCount;
    maxiteration = gaopt.MaxIteration;
    verbose = gaopt.Verbose;
    maxstall = gaopt.Maxstall;
    tolerance = gaopt.Tolerance;
    creationfunction = gaopt.Creationfunction;
    fitnessscalefunction = gaopt.Fitnessscalefunction;
    distancefunction = gaopt.Distancefunction;
    selectfunction = gaopt.Selectfunction;
    crossoverfunction = gaopt.Crossoverfunction;
    mutationfunction = gaopt.Mutationfunction;
    hyperfunction = gaopt.Hyperfunction;
    isMultiobjectives = targetsize>1;

    % Initate GA Data-Structure
    ga = struct(...
        'isMultiobjectives',isMultiobjectives,...
        'Populations',zeros(populationsize,featuresize),...
        'Fitness',zeros(populationsize,targetsize),...
        'Objectives',zeros(populationsize,targetsize),...
        'Bestindividual',zeros(1,featuresize),...
        'Bestfitness',zeros(1,targetsize),...
        'Bestobjective',zeros(1,targetsize),...
        'Ranks',zeros(1,populationsize),...
        'Distances',zeros(1,populationsize),...
        'ParetoFrontindividuals',[],...
        'ParetoFrontfitness',[],...
        'ParetoFrontobjectives',[]...
    );
    ga.Populations = creationfunction(gaopt);
    if ~isempty(initialpopulations)
        ga.Populations = initialpopulations;
    end
    stallobjective = min(ga.Objectives); stallcount = 0;
    ga.Objectives = func(ga.Populations);
    ga.Fitness = ga.Objectives + constraintfunction(ga.Populations);

    % Initiate for single-objective
    [value,index] = min(ga.Fitness);
    ga.Bestindividual = ga.Populations(index,:);
    ga.Bestfitness = ga.Fitness(index);
    ga.Bestobjective = ga.Objectives(index);

    % Main Loop
    iter = 0;
    while iter < maxiteration
        if ~isMultiobjectives
            % Elite Operation
            scaledfitness = fitnessscalefunction(ga.Fitness);
            [values,indexs] = sort(scaledfitness,'descend');
            elites = ga.Populations(indexs(1:elitecount),:);  

            % Crossover Operation
            crossovers =  crossoverfunction(ga,gaopt);

            % Mutation Operation
            mutations = mutationfunction(ga,gaopt);
       
            ga.Populations = [elites;crossovers;mutations];

             % Update ga Structure
            ga.Objectives = func(ga.Populations);
            ga.Fitness = ga.Objectives + constraintfunction(ga.Populations);
            [value,index] = min(ga.Fitness);

            % Update Best Point
            if value < ga.Bestfitness
                ga.Bestindividual = ga.Populations(index,:);
                ga.Bestfitness = ga.Fitness(index);
                ga.Bestobjective = ga.Objectives(index);
            end

            if verbose
                fprintf('%dth updating with best objective %f...\n',iter+1,ga.Bestobjective);
            end

            iter = iter + 1;
            hyperfunction(ga,gaopt);

            % Check Stall
            if (ga.Bestobjective - stallobjective)/stallobjective < tolerance
                stallcount = stallcount + 1;
                if stallcount > maxstall
                    if verbose
                        fprintf('Terminate: Max Stall\n');
                    end
                    return;
                end
            else
                stallcount = 0;
                stallobjective = ga.Bestobjective;
            end

        else
            % Crossover and Mutation
            scaledfitness = fitnessscalefunction(ga.Fitness);
            [ga.Ranks,ga.Distances] = distancefunction(scaledfitness);

            crossovers =  crossoverfunction(ga,gaopt);
            mutations = mutationfunction(ga,gaopt);
            populations = [crossovers;mutations];
            objectives = func(populations);
            fitness = objectives + constraintfunction(populations);

            % Rank populations
            paretonum = size(ga.ParetoFrontindividuals,1);
            totalpopulations = [ga.Populations; populations; ga.ParetoFrontindividuals];
            totalfitness = [ga.Fitness; fitness; ga.ParetoFrontfitness];
            totalobjectives = [ga.Objectives; objectives; ga.ParetoFrontobjectives];
            scaledfitness = fitnessscalefunction(totalfitness);
            [ranks,distances] = distancefunction(scaledfitness);
            [values,indexs] = sortrows([ranks,-distances]);

            % Update Pareto Forntier
            pos = find(ranks==0);
            ga.ParetoFrontindividuals = totalpopulations(pos,:);
            ga.ParetoFrontfitness = totalfitness(pos,:);
            ga.ParetoFrontobjectives = totalobjectives(pos,:);

            % Update GA populations
            totallength = size(totalpopulations,1);
            [values,indexs] = sortrows([ranks(1:(totallength-paretonum)),-distances(1:(totallength-paretonum))]);
            pos = indexs(1:populationsize);
            ga.Populations = totalpopulations(pos,:);
            ga.Fitness = totalfitness(pos,:);
            ga.Objectives = totalobjectives(pos,:);

            if verbose
                fprintf('%dth updating with number of Pareto fronts %d...\n',iter+1,size(ga.ParetoFrontindividuals,1));
            end
            iter = iter + 1;
            hyperfunction(ga,gaopt);
        end

       
    end
    if verbose
        fprintf('Terminate: Max Iterations\n');
    end
end