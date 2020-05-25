function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    y_predicted = theta(1)*X(:,1) + theta(2)*X(:,2);  %computing the predicted y with current theta values
    derivative_theta0 = -(1/m)*sum(y_predicted-y);  %define the derivative part for equation theta 0
    derivative_theta1 = -(1/m)*sum(X(:,2)'*(y_predicted-y)); %define the derivative part for equation theta 0
    theta(1) = theta(1)+(alpha*derivative_theta0); %theta 0 assignment
    theta(2) = theta(2)+(alpha*derivative_theta1); %theta 1 assignment
##    y_predicted = (X*theta); %computing the predicted y with current theta values
##    derivative_theta = -(1/m)*(X'*(y_predicted-y)); %define the derivative part for equation theta 0
##    theta = theta+(alpha*derivative_theta); %theta assignment






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
