%{
The provided function takes in a vector x of length 3 
and returns a scalar objective f to be minimized, 
a vector of constraints c to be enforced as c <= 0, 
a vector gf containing the gradients of f, 
and a matrix gc containing the gradients of the constraints 
(gc[i, j] = partial c_i / partial x_j).  
In addition to the nonlinear constraints, you should also 
enforce the constraint that x >= 0.  
The deterministic optimum of this problem is:
x^* = [0.06293586, 2.91716569, 0.07491246], f^*=1.158963

Using the simplified reliability-based optimization methodology 
discussed in the text, find the reliable optimum under the 
assumption that the inputs are normally distributed and have 
a standard deviation of 
sigma_1 = 0.01, sigma_2 = 0.1, sigma_3 = 0.05.  
Each constraint should be satisfied with a target 
reliability of 99.5%.  
Briefly describe the how/why of your approach 
and turn in your code.
%}


function [f, c, gf, gc] = prob(x)


f = -x(1)*x(2)*x(3) + (x(2) - 4)^2 + 10*x(1)^3*x(3);

c = zeros(2, 1);
c(1) = 2*x(1)^2 + x(2) + x(3) - 3.0;
c(2) = -2*x(1) - 4*x(2)*x(3) + 1.0;

gf = zeros(3, 1);
gf(1) = -x(2)*x(3) + 30*x(1)^2*x(3);
gf(2) = -x(1)*x(3) + 2*(x(2) - 4);
gf(3) = -x(1)*x(2) + 10*x(1)^3;

gc = zeros(2, 3);

gc(1, 1) = 4*x(1);
gc(1, 2) = 1.0;
gc(1, 3) = 1.0;

gc(2, 1) = -2.0;
gc(2, 2) = -4*x(3);
gc(2, 3) = -4*x(2);


end