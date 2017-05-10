function [ranks,distance] = psoPareto(fitness)
    [popsize,targetsize] = size(fitness);
    ranks = zeros(1,popsize);
    distance = zeros(1,popsize);
    
    fronts = {};
    fronts{1} = [];
    
    dominatedindividuals = {};
    dominatecount = zeros(1,popsize);
    
    for i = 1:popsize
        dominatedindividuals{i} = [];
        for j = 1:popsize
            if (sum(fitness(i,:) < fitness(j,:)) == 0)&&(sum(fitness(i,:) == fitness(j,:)) < targetsize)
                dominatedindividuals{i} =  [dominatedindividuals{i},j];
            else
                if (sum(fitness(i,:) > fitness(j,:)) == 0)&&(sum(fitness(i,:) == fitness(j,:)) < targetsize)
                    dominatecount(i) = dominatecount(i) + 1;
                end
            end
        end
        if dominatecount(i) == 0
            fronts{1} = [fronts{1},i];
            ranks(i) = 0;
        end        
    end
    
    index = 1;
    while length(fronts{index}) > 0
        nextfront = [];
        for i = fronts{index}
            for j = dominatedindividuals{i}
                dominatecount(j) = dominatecount(j) - 1;
                if dominatecount(j) == 0
                    ranks(j) = index + 1;
                    nextfront = [nextfront,j];
                end
            end
        end
        index = index + 1;
        fronts = [fronts,nextfront];

        if index > length(fronts)
            break;
        end
    end
    
    if length(fronts) > 0
        for index = 1:length(fronts)
            front = fronts{index};
            frontnum = length(front);
            if frontnum > 0
                for k = 1:targetsize
                    frontfitness = fitness(front,k);
                    [sortvalue,sortindex] = sort(frontfitness);
                    maxfrontfitness = sortvalue(frontnum);
                    minfrontfitness = sortvalue(1);
                    posmax = front(sortindex(frontnum));
                    posmin = front(sortindex(1));
                    
                    infinit = 100*maxfrontfitness;  % This value shall be the maximum of objective functions
                    distance(posmax) = infinit;
                    distance(posmin) = infinit;
                    
                    for i = 2:frontnum-1
                        pos = front(sortindex(i));
                        left = front(sortindex(i-1));
                        right = front(sortindex(i+1));
                        if maxfrontfitness == minfrontfitness
                            distance(pos) = infinit;
                        else
                            distance(pos) = (distance(right)-distance(left))/(maxfrontfitness - minfrontfitness);
                        end
                    end
                end
            end
        end
    end
end