% Author: Yang Long
%
% E-mail: longyang_123@yeah.net

function scalefitness = psoFitnessScale(fitness)
    [particlesize,targetsize] = size(fitness);
    scalefitness = zeros(particlesize,targetsize);
    [values,indexs] = sort(fitness);
    for i = 1:targetsize
        scalefitness(indexs(:,i),i) = 1./((1:particlesize).^0.5);
    end
end