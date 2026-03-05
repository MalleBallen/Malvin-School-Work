# benders_battery_schedule.py
# Task 7 – Benders’ decomposition for two-stage stochastic battery scheduling (Pyomo + GLPK)
# Model and data are taken from "Optimization Project.pdf" (MOD500 course assignment). 
# Author: Malvin Varpe and Nicolai Rhode Garder
#
# Outputs:
#   - results_first_stage_x.csv
#   - results_second_stage.csv
#   - convergence.csv
#!! Important !!:
# To run from your computer, you need to have Pyomo and GLPK installed. Change the path to 
# glpsol_path to point to your local glpsol executable. 
# Note:
# I am using the python codes from the course materials as inspiration in this python file

import math                    # Standard library: used for +/− infinity and math helpers
import pandas as pd            # For writing tabular results (CSV files)
import pyomo.environ as pyo    # Pyomo modeling framework (sets, vars, objectives, constraints, solvers)
import matplotlib.pyplot as plt # For plotting convergence graphs

# Needed to specify path to glpsol for it to work
glpsol_path = r"C:\Users\malvi\OneDrive\Skole\UiS\MOD500\glpk-4.65\w64\glpsol.exe"



# -----------------------------
# 0) Data (from the assignment)
# -----------------------------
def build_data():
    T = list(range(1, 25))  # hours 1..24 - time periods for the day
    S = [1, 2, 3]           
    prob = {1: 0.3, 2: 0.4, 3: 0.3}  

    # c_t (NOK/MWh) for t=1..24  [Table 1] 
    c = {
        1:760, 2:740, 3:720, 4:710, 5:720, 6:770, 7:830, 8:900, 9:940, 10:920, 11:900, 12:880,
        13:870, 14:860, 15:860, 16:880, 17:930, 18:990, 19:1010, 20:990, 21:950, 22:900, 23:860, 24:820
    }

    # Demand D_{s,t} (MWh)  [Table 2] 
    D = {
        1: { 1:266.0, 2:256.5, 3:247.0, 4:242.3, 5:247.0, 6:275.5, 7:313.5, 8:342.0, 9:361.0, 10:351.5, 11:346.8, 12:337.3, 13:332.5, 14:327.8, 15:332.5, 16:351.5, 17:380.0, 18:399.0, 19:389.5, 20:370.5, 21:342.0, 22:313.5, 23:294.5, 24:275.5 },
        2: { 1:280,   2:270,   3:260,   4:255,   5:260,   6:290,   7:330,   8:360,   9:380,   10:370,   11:365,   12:355,   13:350,   14:345,   15:350,   16:370,   17:400,   18:420,   19:410,   20:390,   21:360,   22:330,   23:310,   24:290 },
        3: { 1:302.4, 2:291.6, 3:280.8, 4:275.4, 5:280.8, 6:313.2, 7:356.4, 8:388.8, 9:410.4, 10:399.6, 11:394.2, 12:383.4, 13:378.0, 14:372.6, 15:378.0, 16:399.6, 17:432.0, 18:453.6, 19:442.8, 20:421.2, 21:388.8, 22:356.4, 23:334.8, 24:313.2 }
    }

    # Wind W_{s,t} (MWh)  [Table 3] 
    W = {
        1: { 1:93.1, 2:103.0, 3:124.7, 4:141.7, 5:165.8, 6:185.4, 7:207.8, 8:225.2, 9:207.2, 10:184.1, 11:161.4, 12:147.8, 13:136.6, 14:128.8, 15:120.1, 16:130.8, 17:146.8, 18:163.1, 19:177.7, 20:169.0, 21:146.8, 22:121.5, 23:103.4, 24:97.3 },
        2: { 1:118.8, 2:131.7, 3:159.0, 4:181.6, 5:212.0, 6:236.9, 7:265.8, 8:287.7, 9:264.0, 10:234.9, 11:206.9, 12:189.3, 13:174.7, 14:164.0, 15:153.9, 16:168.7, 17:189.4, 18:210.7, 19:229.5, 20:218.4, 21:189.5, 22:156.3, 23:133.1, 24:125.2 },
        3: { 1:148.5, 2:164.6, 3:199.1, 4:227.0, 5:265.0, 6:296.1, 7:332.3, 8:359.6, 9:330.1, 10:293.7, 11:258.6, 12:236.6, 13:218.4, 14:205.0, 15:192.4, 16:210.9, 17:236.7, 18:263.4, 19:287.0, 20:273.0, 21:236.8, 22:195.4, 23:166.4, 24:156.5 }
    }

    # Real-time market price p_{s,t} (NOK/MWh)  [Table 4] 
    p = {
        1: { 1:837.0, 2:818.4, 3:799.8, 4:790.5, 5:799.8, 6:837.0, 7:911.4, 8:967.2, 9:1023.0, 10:1004.4, 11:985.8, 12:967.2, 13:950.6, 14:930.0, 15:930.0, 16:950.6, 17:1004.4, 18:1078.8, 19:1097.4, 20:1078.8, 21:1023.0, 22:967.2, 23:911.4, 24:874.2 },
        2: { 1:900,   2:880,   3:860,   4:850,   5:860,   6:900,   7:980,   8:1040,  9:1100,   10:1080,  11:1060,  12:1040,  13:1020,  14:1000,  15:1000,  16:1020,  17:1080,  18:1160,  19:1180,  20:1160,  21:1100,  22:1040,  23:980,   24:940 },
        3: { 1:990.0, 2:968.0, 3:946.0, 4:935.0, 5:946.0, 6:990.0, 7:1078.0, 8:1144.0, 9:1210.0, 10:1188.0, 11:1166.0, 12:1144.0, 13:1122.0, 14:1100.0, 15:1100.0, 16:1122.0, 17:1188.0, 18:1276.0, 19:1298.0, 20:1276.0, 21:1210.0, 22:1144.0, 23:1078.0, 24:1034.0 }
    }

    # Market purchase capacity M_cap,t (MWh)  [Table 5] 
    Mcap = {
        1:140,2:140,3:140,4:140,5:150,6:170,7:190,8:200,9:200,10:190,11:180,12:170,
        13:160,14:160,15:160,16:170,17:152,18:168,19:168,20:160,21:180,22:160,23:150,24:145
    }

    # Constants (penalties, efficiencies, limits)
    P_unmet = 100000.0  # NOK/MWh - very high penalty per unmet demand
    k = 320.0           # NOK/MWh - curtailment penalty
    Emax = 120.0        # MWh - upper bound for battery charge
    Pmax_ch = 45.0      # MW  - max charge power (1hr eq MWh)
    Pmax_dis = 55.0     # MW  - max discharge power
    eta_ch = 0.95       # charge efficiency
    eta_dis = 0.95      # discharge efficiency
    S0 = 60.0           # MWh - initial battery charge at t=1 
    Smin = 55.0         # MWh - battery lower bound
    Smax = 85.0         # MWh - battery upper bound
    beta = 12.0         # throughput cost per MWh charged/discharged

    # Upper bound on day-ahead decision x_t, Xmax[2] is max in hour 2 f ex.
    Xmax = {t: D[3][t] for t in T}

    # Bundle everything into a dictionary for convenience
    data = dict(T=T, S=S, prob=prob, c=c, D=D, W=W, p=p, Mcap=Mcap,
                k=k, P_unmet=P_unmet, Emax=Emax, Pmax_ch=Pmax_ch, Pmax_dis=Pmax_dis,
                eta_ch=eta_ch, eta_dis=eta_dis, S0=S0, Smin=Smin, Smax=Smax,
                beta=beta, Xmax=Xmax)
    return data  


# ---------------------------------------------
# 1) Master Problem (first-stage + alpha)
# ---------------------------------------------
def build_master(data):
    m = pyo.ConcreteModel(name="Master")  # Instantiate a concrete Pyomo model

    T = data['T']          # time periods
    c = data['c']          # day-ahead prices
    Xmax = data['Xmax']    # per-hour upper bounds on x

    m.T = pyo.Set(initialize=T)  # Index time hours

    # First-stage decision: x_t (0 <= x_t <= Xmax_t)
    # Using a lambda to supply per-index bounds based on Xmax[t]
    m.x = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=lambda _m, t: (0.0, Xmax[t]))

    # alpha: lower bound on expected second-stage cost (>= 0 since cost components are >= 0)
    m.alpha = pyo.Var(domain=pyo.NonNegativeReals)

    # Objective: first-stage purchase cost + alpha 
    def obj_rule(_m):
        return sum(c[t]*_m.x[t] for t in _m.T) + _m.alpha
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Container to add Benders cuts iteratively
    m.cut_index = pyo.Set(initialize=[], ordered=True)  # index of cuts, for graphing
    m.cuts = pyo.ConstraintList()                       # we will append cuts here

    return m  


# -------------------------------------------------
# 2) Subproblem (for one scenario, fixed x = xhat)
# -------------------------------------------------
def build_subproblem(data, s, xhat):
    sp = pyo.ConcreteModel(name=f"Subproblem_s{s}")  # Per scenario 

    # Unpack data and constants for readability
    T = data['T']
    D = data['D'][s]       # demands for scenario s
    W = data['W'][s]       # winds for scenario s
    p = data['p'][s]       # purchase prices for scenario s
    Mcap = data['Mcap']
    k = data['k']
    P_unmet = data['P_unmet']
    Emax = data['Emax']
    Pmax_ch = data['Pmax_ch']
    Pmax_dis = data['Pmax_dis']
    eta_ch = data['eta_ch']
    eta_dis = data['eta_dis']
    S0 = data['S0']
    Smin = data['Smin']
    Smax = data['Smax']
    beta = data['beta']

    sp.T = pyo.Set(initialize=T)  # scenario subproblem has same time set

    # Fixed day ahead schedule xhat[t] 
    sp.xhat = pyo.Param(sp.T, initialize=lambda _m, t: xhat[t], within=pyo.NonNegativeReals)

    # Second-stage variables (all >= 0)
    sp.q = pyo.Var(sp.T, domain=pyo.NonNegativeReals, bounds=(0.0, Pmax_ch))      # charge power
    sp.r = pyo.Var(sp.T, domain=pyo.NonNegativeReals, bounds=(0.0, Pmax_dis))     # discharge power
    sp.m = pyo.Var(sp.T, domain=pyo.NonNegativeReals, bounds=lambda _m, t: (0.0, Mcap[t]))  # market purchase
    sp.u = pyo.Var(sp.T, domain=pyo.NonNegativeReals)                              # wind curtailment
    sp.l = pyo.Var(sp.T, domain=pyo.NonNegativeReals)                              # unmet demand slack 
    sp.S = pyo.Var(sp.T, domain=pyo.NonNegativeReals, bounds=(0.0, Emax))          # state of charge

    # Battery dynamics. SOC - state of charge
    def soc_first_rule(_m):
        # t=1 update: S1 = S0 + eta_ch*q1 - (1/eta_dis)*r1
        return _m.S[1] == S0 + eta_ch*_m.q[1] - (1.0/eta_dis)*_m.r[1]
    sp.SOC1 = pyo.Constraint(rule=soc_first_rule)

    def soc_rule(_m, t):
        if t == 1:
            return pyo.Constraint.Skip  # already enforced by SOC1
        # For t>1: St = S(t-1) + eta_ch*q_t - (1/eta_dis)*r_t
        return _m.S[t] == _m.S[t-1] + eta_ch*_m.q[t] - (1.0/eta_dis)*_m.r[t]
    sp.SOC = pyo.Constraint(sp.T, rule=soc_rule)

    # Terminal SOC band: enforce end-of-day SOC within bounds
    def terminal_rule(_m):
        return pyo.inequality(Smin, _m.S[max(_m.T)], Smax)
    sp.Terminal = pyo.Constraint(rule=terminal_rule)

    # Supply–demand balance (stage-linking constraints)
    # xhat[t] + r + m + (W - u) + l = D + q
    def balance_rule(_m, t):
        return _m.xhat[t] + _m.r[t] + _m.m[t] + (W[t] - _m.u[t]) + _m.l[t] == D[t] + _m.q[t]
    sp.balance = pyo.Constraint(sp.T, rule=balance_rule)

    # Objective: scenario cost to minimize 
    def obj_rule(_m):
        purchase = sum(p[t]*_m.m[t] for t in _m.T)
        curtail = sum(k*_m.u[t] for t in _m.T)
        unmet = sum(P_unmet*_m.l[t] for t in _m.T)
        throughput = sum(beta*(_m.q[t] + _m.r[t]) for t in _m.T)
        return purchase + curtail + unmet + throughput
    sp.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Duals (to extract alpha hat for the balance constraints after solving)
    sp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)  # tells Pyomo to import dual prices from solver

    return sp  # return the constructed per-scenario LP


# ---------------------------------------------------------
# 3) Solve a scenario subproblem & extract obj and duals
# ---------------------------------------------------------
def solve_subproblem(sp, solver='glpk'):
    opt = pyo.SolverFactory(solver, executable=glpsol_path)  # configure solver (GLPK, with correct path to my location)
    result = opt.solve(sp, tee=False)                        # solve silently (tee=True for logs)

    # Gather objective value (recourse cost for this scenario at xhat)
    obj = pyo.value(sp.OBJ)

    # Duals for balance constraints: lambda_t 
    lambda_t = {t: sp.dual[sp.balance[t]] for t in sp.T}  # Pyomo stores duals keyed by constraint
    return obj, lambda_t  # will be used to form Benders cuts


# -------------------------------------------------------------------
# 4) Add aggregated Benders optimality cut to the master problem
#    Using subgradient g_{s,t} = -lambda_{s,t}, with cut at xhat
#    alpha  >=  Σ_s pi_s [ alphahat_s(xhat) + Σ_t g_{s,t} (x_t - xhat_t) ]
# -------------------------------------------------------------------
def add_benders_cut(master, data, xhat, scen_objs, scen_duals, cut_id):
    prob = data['prob']   # scenario probabilities
    T = data['T']         # time set

    # Build the term of the cut:
    # constant = Σ_s p_s [ obj_s + Σ_t g_{s,t}*(-xhat_t) ]
    # slope    = Σ_s p_s Σ_t g_{s,t} * x_t
    const_term = 0.0
    slope_terms = []
    for s in data['S']: #Summing up for all probabilities to find the expected value of the optimal solution
        obj_s = scen_objs[s]   # alphahat_s(xhat)
        lambda_s = scen_duals[s]   # lambda_{s,t} duals
        
        g = {t: -lambda_s[t] for t in T}
        # accumulate the constant part (at xhat)
        const_term += prob[s] * (obj_s + sum(g[t]*(-xhat[t]) for t in T))
        # accumulate slope terms multiplying the master vars x[t]
        for t in T:
            slope_terms.append(prob[s] * g[t] * master.x[t])

    # Final cut: alpha >= constant + Σ slope_terms
    cut_expr = master.alpha >= const_term + sum(slope_terms)
    master.cuts.add(cut_expr)   # append to ConstraintList
    master.cut_index.add(cut_id)  # The bookkeeping index


# --------------------------------------
# 5) Driver: Benders loop (outer solve)
# --------------------------------------
def benders_solve(data, max_iters=100, tol_abs=1e-4, tol_rel=1e-5, solver='glpk'):
    master = build_master(data)                                     # build master model
    opt = pyo.SolverFactory(solver, executable=glpsol_path)         

    # Convergence trackers
    history = []              # to record LB, UB, gaps per iteration for CSV
    LB = -math.inf            # global lower bound 
    UB = math.inf             # global upper bound 

    # Initial xhat comes from solving the master with no cuts (alpha free to be 0)
    for it in range(1, max_iters+1):
        # Solve master with current cuts (first iter has none)
        res_m = opt.solve(master, tee=False)
        xhat = {t: pyo.value(master.x[t]) for t in master.T}  # current day-ahead plan
        alpha_hat = pyo.value(master.alpha)                   # current alpha value (LB on recourse)
        first_stage_cost = sum(data['c'][t]*xhat[t] for t in master.T)  # ∑ c_t xhat_t

        # Solve all scenario subproblems at xhat
        scen_objs = {}     # alphahat_s (xhat) per scenario
        scen_duals = {}    # lambda_{s,t} per scenario
        expected_recourse = 0.0
        for s in data['S']:
            sp = build_subproblem(data, s, xhat)                 # construct scenario LP with xhat
            obj_s, pi_s = solve_subproblem(sp, solver=solver)    # solve and extract duals
            scen_objs[s] = obj_s
            scen_duals[s] = pi_s
            expected_recourse += data['prob'][s] * obj_s         # accumulate expectation

        # Upper bound: feasible combined objective at xhat
        UB = min(UB, first_stage_cost + expected_recourse)

        # Lower bound: master objective value (∑ c_t x_t + alpha with cuts) after solve
        LB = pyo.value(master.OBJ)

        # Gap checks (absolute and relative)
        gap_abs = UB - LB
        gap_rel = gap_abs / max(1.0, abs(UB))  # avoid divide-by-zero
        history.append({'iter': it, 'LB': LB, 'UB': UB, 'gap_abs': gap_abs, 'gap_rel': gap_rel})

        # Convergence criterion: stop if gap small enough
        if gap_abs <= tol_abs or gap_rel <= tol_rel:
            break  # converged

        # Add aggregated optimality cut at the current xhat and loop again
        add_benders_cut(master, data, xhat, scen_objs, scen_duals, cut_id=it)

    # Final x*: read optimal/master solution after last iteration
    x_star = {t: pyo.value(master.x[t]) for t in master.T}

    # Re-solve subproblems at x* to collect detailed second-stage variables for output CSV
    second_stage_rows = []
    for s in data['S']:
        sp = build_subproblem(data, s, x_star)                                   # build with x*
        _ = pyo.SolverFactory(solver, executable=glpsol_path).solve(sp, tee=False)  # solve sp

        for t in sp.T:
            # Record per-hour, per-scenario ops and SOC
            second_stage_rows.append({
                'scenario': s,
                'hour': t,
                'x': x_star[t],
                'q': pyo.value(sp.q[t]),
                'r': pyo.value(sp.r[t]),
                'm': pyo.value(sp.m[t]),
                'u': pyo.value(sp.u[t]),
                'l': pyo.value(sp.l[t]),
                'S': pyo.value(sp.S[t]),
            })

    # Save CSVs required by Task 7 (first-stage, second-stage, and convergence history)
    pd.DataFrame({'hour': list(x_star.keys()),
                  'x': list(x_star.values())}).to_csv('results_first_stage_x.csv', index=False)

    pd.DataFrame(second_stage_rows).to_csv('results_second_stage.csv', index=False)
    pd.DataFrame(history).to_csv('convergence.csv', index=False)

    return {
        'x_star': x_star,  # optimal day-ahead plan
        'LB': LB,          # lower bound at termination
        'UB': UB,          # upper bound at termination
        'history': history # iteration log
    }

def plot_convergence(filepath='convergence.csv'):
    df = pd.read_csv(filepath)

    # Plot LB, UB, and gap_abs over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(df['iter'], df['LB'], marker='o', label='Lower Bound (LB)')
    plt.plot(df['iter'], df['UB'], marker='o', label='Upper Bound (UB)')
    plt.plot(df['iter'], df['gap_abs'], marker='o', linestyle='--', label='Absolute Gap')

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value / Gap')
    plt.title('Benders Decomposition Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('benders_convergence.png')
    plt.show()

if __name__ == "__main__":
    data = build_data()  # assemble all inputs
    results = benders_solve(data, max_iters=100, tol_abs=1e-4, tol_rel=1e-5, solver='glpk')
    # Run Benders loop with reasonable tolerances.

    print("\nOptimal day-ahead schedule x* (first 10 entries):")
    for t in sorted(results['x_star'].keys())[:10]:
        # Print the first 10 hours for quick inspection
        print(f"  t={t:2d}  x*={results['x_star'][t]:.4f}")

    print("\nCSV files written: results_first_stage_x.csv, results_second_stage.csv, convergence.csv")

    plot_convergence()


