function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X(1,:))(2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Calc J
J = 0;
for i=1:m
  hval = sigmoid(theta' * X(i,:)');
  J = J + -1 * y(i)*log(hval) - (1 - y(i)) * log(1 - hval);
end;
J = J / m;

reg = 0;
for j=2:n
  reg = reg + theta(j) * theta(j); 
end;
J = J + (lambda / (2 * m)) * reg;

grad = zeros(size(theta));
for i=1:m
  hval = sigmoid(theta' * X(i,:)');
  grad = grad + (hval - y(i)) * X(i,:)';
end;
grad = grad / m;
for j=2:n
    grad(j) = grad(j) + (lambda / m) * theta(j);
end;




% =============================================================

end
