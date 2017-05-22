function scaledfitness = gaFitnessScale(fitness)
    [populationsize,targetsize] = size(fitness);
    scaledfitness = zeros(populationsize,targetsize);
    [values,indexs] = sort(fitness);
    for i = 1:targetsize
        scaledfitness(indexs(:,i),i) = 1./((1:populationsize).^0.5);
    end
end