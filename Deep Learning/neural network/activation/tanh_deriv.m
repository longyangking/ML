function result = tanh_deriv(x)
    result = 1 - tanh(x).^2;
end