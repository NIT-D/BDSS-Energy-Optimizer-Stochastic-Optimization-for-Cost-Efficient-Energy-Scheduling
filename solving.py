# solving.py
from pyomo.environ import *

def build_stochastic_model(
    p_base,
    p_scenarios,
    prob_scenarios,
    total_demand,
    x_min,
    x_max,
    penalty=1.0,
    max_ramp=2.0,
    risk_weight=0.1,
    robust_weight=0.05,
    real_time_flexibility=0.5,
    base_coverage_target=0.9,          # % of demand that base must cover
    price_sensitivity=0.3,             # adjustable price sensitivity
    cvar_alpha=0.95                    # adjustable CVaR confidence level
):
    model = ConcreteModel()

    # Sets
    model.H = RangeSet(0, 23)
    model.S = RangeSet(0, len(p_scenarios) - 1)

    # Parameters
    model.p = {h: p_base[h] for h in model.H}
    model.p_s = {(s, h): p_scenarios[s][h] for s in model.S for h in model.H}
    model.prob = {s: prob_scenarios[s] for s in model.S}

    # Variables
    model.x = Var(model.H, domain=NonNegativeReals, bounds=(x_min, x_max))
    model.x_s = Var(model.S, model.H, domain=NonNegativeReals, bounds=(x_min, x_max))
    model.var_alpha = Var(domain=Reals)
    model.cvar_deviation = Var(model.S, domain=NonNegativeReals)
    model.worst_case_cost = Var(domain=NonNegativeReals)
    model.real_time_adjustment = Var(model.S, model.H, domain=Reals, bounds=(-real_time_flexibility, real_time_flexibility))
    model.smooth_penalty = Var(RangeSet(1, 23), domain=NonNegativeReals)

    # Objective Function
    def objective_rule(m):
        base_cost = summation(m.p, m.x)
        scenario_costs = sum(
            m.prob[s] * sum(m.p_s[s, h] * (m.x_s[s, h] + m.real_time_adjustment[s, h]) for h in m.H)
            for s in m.S
        )
        price_penalty = 0.1 * sum(m.x[h] * m.p[h] * m.p[h] for h in m.H)
        base_incentive = 0.05 * sum(m.x[h] for h in m.H)

        cvar_penalty = (risk_weight * 0.1) * (
            m.var_alpha + (1 / (1 - cvar_alpha)) * sum(m.prob[s] * m.cvar_deviation[s] for s in m.S)
        )
        robust_penalty = (robust_weight * 0.1) * m.worst_case_cost
        smoothness = (penalty * 0.1) * sum(m.smooth_penalty[h] for h in range(1, 24))
        scenario_adjustment_penalty = 0.02 * sum(m.x_s[s, h] for s in m.S for h in m.H)

        return (
            base_cost + scenario_costs + price_penalty
            - base_incentive + cvar_penalty + robust_penalty
            + smoothness + scenario_adjustment_penalty
        )

    model.obj = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    def demand_rule(m, s):
        avg_scenario_price = sum(m.p_s[s, h] for h in m.H) / 24
        avg_base_price = sum(m.p[h] for h in m.H) / 24
        price_ratio = avg_scenario_price / avg_base_price
        demand_adjustment = total_demand * price_sensitivity * (1 - price_ratio)
        scenario_demand = total_demand + demand_adjustment
        return sum(m.x[h] + m.x_s[s, h] for h in m.H) == scenario_demand

    def hourly_bounds_rule(m, s, h):
        return (x_min, m.x[h] + m.x_s[s, h], x_max)

    def min_usage_rule(m, h):
        return m.x[h] >= x_min

    def cvar_rule(m, s):
        scenario_cost = sum(
            m.p_s[s, h] * (m.x[h] + m.x_s[s, h] + m.real_time_adjustment[s, h])
            for h in m.H
        )
        return scenario_cost - m.var_alpha <= m.cvar_deviation[s]

    def worst_case_rule(m, s):
        scenario_cost = sum(
            m.p_s[s, h] * (m.x[h] + m.x_s[s, h] + m.real_time_adjustment[s, h])
            for h in m.H
        )
        return scenario_cost <= m.worst_case_cost

    def real_time_lower_rule(m, s):
        return sum(m.real_time_adjustment[s, h] for h in m.H) >= -0.1 * total_demand

    def real_time_upper_rule(m, s):
        return sum(m.real_time_adjustment[s, h] for h in m.H) <= 0.1 * total_demand

    def smooth_pos_rule(m, h):
        return m.smooth_penalty[h] >= m.x[h] - m.x[h - 1]

    def smooth_neg_rule(m, h):
        return m.smooth_penalty[h] >= -(m.x[h] - m.x[h - 1])

    def ramp_up_rule(m, h):
        if h > 0:
            return m.x[h] - m.x[h - 1] <= max_ramp
        return Constraint.Skip

    def ramp_down_rule(m, h):
        if h > 0:
            return m.x[h - 1] - m.x[h] <= max_ramp
        return Constraint.Skip

    # Base coverage (% of demand covered by base schedule)
    def base_coverage_rule(m):
        return sum(m.x[h] for h in m.H) >= base_coverage_target * total_demand

    # Add constraints
    model.demand = Constraint(model.S, rule=demand_rule)
    model.hourly_bounds = Constraint(model.S, model.H, rule=hourly_bounds_rule)
    model.min_usage = Constraint(model.H, rule=min_usage_rule)
    model.cvar = Constraint(model.S, rule=cvar_rule)
    model.worst_case = Constraint(model.S, rule=worst_case_rule)
    model.real_time_lower = Constraint(model.S, rule=real_time_lower_rule)
    model.real_time_upper = Constraint(model.S, rule=real_time_upper_rule)
    model.smooth_pos = Constraint(RangeSet(1, 23), rule=smooth_pos_rule)
    model.smooth_neg = Constraint(RangeSet(1, 23), rule=smooth_neg_rule)
    model.ramp_up = Constraint(model.H, rule=ramp_up_rule)
    model.ramp_down = Constraint(model.H, rule=ramp_down_rule)
    model.base_coverage_con = Constraint(rule=base_coverage_rule)

    return model


def solve_model(model, solver_name="gurobi"):
    solver = SolverFactory(solver_name)
    if not solver.available():
        fallback_solvers = ["cbc", "glpk", "ipopt"]
        for fallback in fallback_solvers:
            fallback_solver = SolverFactory(fallback)
            if fallback_solver.available():
                solver = fallback_solver
                break
        else:
            raise RuntimeError(f"No suitable solver available. Tried: {solver_name}, {', '.join(fallback_solvers)}")

    if hasattr(solver, 'options'):
        solver.options['TimeLimit'] = 300
        solver.options['MIPGap'] = 0.01

    try:
        results = solver.solve(model, tee=False)
        if str(results.solver.termination_condition) == "optimal":
            return results
        elif str(results.solver.termination_condition) == "infeasible":
            raise RuntimeError("Model is infeasible. Check your constraints.")
        elif str(results.solver.termination_condition) == "unbounded":
            raise RuntimeError("Model is unbounded. Check your objective function.")
        else:
            raise RuntimeError(f"Solver terminated with status: {results.solver.termination_condition}")
    except Exception as e:
        raise RuntimeError(f"Solver error: {str(e)}")


def extract_solution(model):
    x = {h: value(model.x[h]) for h in model.H}
    x_s = {
        s: {h: value(model.x_s[s, h]) if model.x_s[s, h].value is not None else 0 for h in model.H}
        for s in model.S
    }
    real_time_adj = {
        s: {h: value(model.real_time_adjustment[s, h]) if model.real_time_adjustment[s, h].value is not None else 0 for h in model.H}
        for s in model.S
    }
    scenario_costs = {
        s: sum(model.p_s[s, h] * (x[h] + x_s[s][h] + real_time_adj[s][h]) for h in model.H)
        for s in model.S
    }
    var_alpha = value(model.var_alpha) if model.var_alpha.value is not None else 0
    cvar_deviation = {
        s: value(model.cvar_deviation[s]) if model.cvar_deviation[s].value is not None else 0
        for s in model.S
    }
    worst_case_cost = value(model.worst_case_cost) if model.worst_case_cost.value is not None else 0

    return {
        "x": x,
        "x_s": x_s,
        "real_time_adjustments": real_time_adj,
        "scenario_costs": scenario_costs,
        "var_alpha": var_alpha,
        "cvar_deviation": cvar_deviation,
        "worst_case_cost": worst_case_cost
    }
