function [xopt, fopt, exitflag, output] = optimize()

    % -------- starting point and bounds ----------
    x0 = [];
    ub = [];
    lb = [];
    % ---------------------------------------------

    % ------ linear constraints ----------------
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    % ------------------------------------------

    % ---- Objective and Constraints -------------
    function [J, c, ceq] = objcon(x)

        % set objective/constraints here
        
    end
    % -------------------------------------------

    % ----------- options ----------------------------
    options = optimoptions('fmincon', ...
        'Algorithm', 'active-set', ...  % choose one of: 'interior-point', 'sqp', 'active-set', 'trust-region-reflective'
        'HonorBounds', 'bounds', ...  % forces optimizer to always satisfy bounds at each iteration
        'Display', 'iter-detailed', ...  % display more information
        'MaxIterations', 1000, ...  % maximum number of iterations
        'MaxFunctionEvaluations', 10000, ...  % maximum number of function calls
        'OptimalityTolerance', 1e-6, ...  % convergence tolerance on first order optimality
        'ConstraintTolerance', 1e-6, ...  % convergence tolerance on constraints
        'FiniteDifferenceType', 'forward', ...  % if finite differencing, can also use central
        'SpecifyObjectiveGradient', false, ...  % supply gradients of objective
        'SpecifyConstraintGradient', false, ...  % supply gradients of constraints
        'CheckGradients', false, ...  % true if you want to check your supplied gradients against finite differencing
        'Diagnostics', 'on');  % display diagnotic information
    % -------------------------------------------


    % -- NOTE: no need to change anything below) --

    % ------- shared variables -----------
    xlast = [];  % last design variables
    Jlast = [];  % last objective
    clast = []; % last nonlinear inequality constraint
    ceqlast = []; % last nonlinear equality constraint
    % --------------------------------------


    % ------ separate obj/con  --------
    function [J] = obj(x)

        % check if computation is necessary
        if ~isequal(x, xlast)
            [Jlast, clast, ceqlast] = objcon(x);
            xlast = x;
        end

        J = Jlast;
    end

    function [c, ceq] = con(x)

        % check if computation is necessary
        if ~isequal(x, xlast)
            [Jlast, clast, ceqlast] = objcon(x);
            xlast = x;
        end

        % set constraints
        c = clast;
        ceq = ceqlast;
    end
    % ------------------------------------------------

    % call fmincon
    [xopt, fopt, exitflag, output] = fmincon(@obj, x0, A, b, Aeq, beq, lb, ub, @con, options);

end
