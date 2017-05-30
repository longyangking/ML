function result = logistic_deriv(x)
    result = logistic(x).*(1 - logistic(x));
end