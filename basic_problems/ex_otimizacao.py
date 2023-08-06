# EXEMPLO DE OTIMIZAÇÃO DE ROTEAMENTO DE FROTAS COM JANELA DE TEMPO
# MATHEUS FREIRE WU JUN/2022

from time import time
import pandas as pd
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pyomo.environ as pyomo

places = {'origem':[0,0],'p1':[50,100],'p2':[-50,150],'p3':[40,-70],'p4':[20,70],'p5':[-50,-40]}
work_time = {'p1':30,'p2':30,'p3':30,'p4':30,'p5':30}
time_window = {'p1':[150,200],'p2':[150,200],'p3':[250,300],'p4':[300,350],'p5':[400,450]}
dist_time = {(p1,p2):math.sqrt((places[p1][0] - places[p2][0])**2 + (places[p1][1] - places[p2][1])**2) for p1 in places for p2 in places if p1 != p2}
transp = {'T1':['m1','m2'],'T2':['m1']}
custo = {'T1':20, 'T2':10}

# PLOT GRAPH
fig1, ax1 = plt.subplots()
for (p1,p2) in dist_time:
    ax1.plot([places[p1][0], places[p2][0]], [places[p1][1], places[p2][1]], linewidth = 0.5, alpha = 0.3)
    if p1 != 'origem':
        ax1.plot(places[p1][0], places[p1][1],'ro')
ax1.plot(places['origem'][0], places['origem'][1],'xb')
plt.show(block = True)


class mip_model:
    def __init__(self):

        self.model = pyomo.ConcreteModel()
        self.var_name = {}
        self.max_time = max([work_time[p] for p in places if p != 'origem']) + max([max(time_window[p]) for p in places if p != 'origem'])

        # PYOMO SETS
        self.places_idx = list(places)
        self.places_idx = pyomo.Set(initialize = self.places_idx)

        self.transp_mach_idx = list((r,m) for r in transp for m in transp[r])
        self.transp_mach_idx = pyomo.Set(initialize = self.transp_mach_idx)

        self.route_idx = [(r,m,p1,p2) for r in transp for m in transp[r] for (p1,p2) in dist_time]
        self.route_idx = pyomo.Set(initialize = self.route_idx)

        self.time_idx = [(r,m,p) for r in transp for m in transp[r] for p in places]
        self.time_idx = pyomo.Set(initialize = self.time_idx)

        # PROBLEM VARIABLES

        # route activation binary variable
        self.model.var_rts_bin = pyomo.Var(self.route_idx, domain=pyomo.Binary)
        self.var_name["var_rts_bin"] = ["transp","mach","p_orig","p_dest"]

        # arrive time
        self.model.var_arr_time = pyomo.Var(self.time_idx)
        self.var_name["var_arr_time"] = ["transp","mach","p_dest"]

        # PROBLEM CONSTRAINTS

        # flux constraint
        def flux_contr(model,r,m,p): # nesse caso, model e i são parametros obrigatorios do pyomo, mas nao utilizados
                return (
                    pyomo.quicksum(self.model.var_rts_bin[(r,m,p1,p)] if (r,m,p1,p) in self.model.var_rts_bin else 0 for p1 in places)
                    -
                    pyomo.quicksum(self.model.var_rts_bin[(r,m,p,p1)] if (r,m,p,p1) in self.model.var_rts_bin else 0 for p1 in places)
                    ==
                    0
                )
        self.model.flux_contr = pyomo.Constraint(self.time_idx, rule=flux_contr)
        print(f' NUMBER OF FLUX CONSTRAINTS: {len(self.model.flux_contr)}')

        # unique entry
        def unique_contr(model, p): # nesse caso, model e i são parametros obrigatorios do pyomo, mas nao utilizados
            if p != 'origem':
                return (
                    pyomo.quicksum(self.model.var_rts_bin[(r,m,p1,p)] if (r,m,p1,p) in self.model.var_rts_bin else 0 for r in transp for m in transp[r] for p1 in places)
                    ==
                    1
                )
            else:
                return pyomo.Constraint.Feasible
        self.model.unique_contr = pyomo.Constraint(self.places_idx, rule=unique_contr)
        print(f' NUMBER OF UNIQUE ENTRY CONSTRAINTS: {len(self.model.unique_contr)}')

        # # unique exit (substitute for multiple start/end calls)
        # def unique_exit(model, r, m): # nesse caso, model e i são parametros obrigatorios do pyomo, mas nao utilizados
        #     return (
        #         pyomo.quicksum(self.model.var_rts_bin[(r,m,p1,'origem')] if (r,m,p1,'origem') in self.model.var_rts_bin else 0 for p1 in self.places_idx)
        #         ==
        #         0
        #     )
        # self.model.unique_exit = pyomo.Constraint(self.transp_mach_idx, rule=unique_exit)
        # print(f' NUMBER OF UNIQUE EXIT CONSTRAINTS: {len(self.model.unique_exit)}')

        # time constraint
        def time_constr(model, r,m,p1,p2): # nesse caso, model e i são parametros obrigatorios do pyomo, mas nao utilizados
            if p1 != 'origem':
                return (
                    self.model.var_arr_time[(r,m,p1)] - self.model.var_arr_time[(r,m,p2)]
                    +
                    (work_time[p1] if p1 != 'origem' else 0)
                    +
                    self.model.var_rts_bin[(r,m,p1,p2)]
                    <=
                    (1 - self.model.var_rts_bin[(r,m,p1,p2)]) * self.max_time
                )
            else:
                return pyomo.Constraint.Feasible
                # return (
                #     self.model.var_arr_time[(r,m,p2)] >= self.model.var_arr_time[(r,m,p1)]
                # )
        self.model.time_constr = pyomo.Constraint(self.route_idx, rule=time_constr)
        print(f' NUMBER OF TIME CONSTRAINTS: {len(self.model.time_constr)}')

        # upper time window
        def up_time_constr(model,r,m,p):
            if p != 'origem':
                return (
                    self.model.var_arr_time[(r,m,p)]
                    <=
                    (1 - pyomo.quicksum(self.model.var_rts_bin[(r,m,p1,p)] if (r,m,p1,p) in self.model.var_rts_bin else 0 for p1 in places))
                    *
                    self.max_time
                    +
                    time_window[p][1]
                )
            else:
                return pyomo.Constraint.Feasible
        self.model.up_time_constr = pyomo.Constraint(self.time_idx, rule=up_time_constr)
        print(f' NUMBER OF UPPER TIME CONSTRAINTS: {len(self.model.up_time_constr)}')

        # lower time window
        def low_time_constr(model,r,m,p):
            if p != 'origem':
                return (
                    self.model.var_arr_time[(r,m,p)]
                    >=
                    pyomo.quicksum(self.model.var_rts_bin[(r,m,p1,p)] if (r,m,p1,p) in self.model.var_rts_bin else 0 for p1 in places)
                    *
                    time_window[p][0]
                )
            else:
                return pyomo.Constraint.Feasible
        self.model.low_time_constr = pyomo.Constraint(self.time_idx, rule=low_time_constr)
        print(f' NUMBER OF UPPER TIME CONSTRAINTS: {len(self.model.low_time_constr)}')

        # PROBLEM FO
        def ObjRule(model):
            return (
                pyomo.quicksum(self.model.var_rts_bin(r,m,p1,p2)*dist_time(p1,p2)*custo[r] for (r,m,p1,p2) in self.model.var_rts_bin)
            )

        self.model.obj = pyomo.Objective(rule = ObjRule) # OBJETIVO do Pyomo

        pass

    def solve_model(self):
        # resolver o modelo e retornar a solulçao
        opt = pyomo.SolverFactory('glpk')
        solution = opt.solve(self.model, load_solutions=False, tee=True)
        print(solution.solver)
        self.model.solutions.load_from(solution) # Passa o valor da solução para as variáveis do modelo ########### <----

    def get_variable(self,varname):
            return getattr(self.model,varname)

    def write_output(self):
        now = datetime.now()
        file = "output_{}.xlsx".format(now.strftime("%y-%m-%d_%H-%M-%S"))
        path = "./"
        output_file_path = os.path.join(path, file)
        writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')
        # Write all variables
        for var in self.var_name:
            # Create basic data_frame
            model_var = self.get_variable(var)
            model_val = model_var.extract_values()
            if self.var_name[var] is None:
                columns=["val"]
                data=[model_val[None]]
            else:
                columns=self.var_name[var]+["val"]
                data=[list(x) + [model_val[x]] for x in model_val]
            df = pd.DataFrame(data=data,columns=columns)
            df = df[df["val"]>0.1]
            df.reset_index(drop=True, inplace=True)
            df.to_excel(writer,var)
        writer.save()

    def plot_results(self):

        model_var = self.get_variable('var_rts_bin')
        model_val = model_var.extract_values()
        columns=self.var_name['var_rts_bin']+["val"]
        data=[list(x) + [model_val[x]] for x in model_val]
        df_rts = pd.DataFrame(data=data,columns=columns)
        df_rts = df_rts[df_rts["val"]>0.1]
        
        model_var = self.get_variable('var_arr_time')
        model_val = model_var.extract_values()
        columns=self.var_name['var_arr_time']+["time"]
        data=[list(x) + [model_val[x]] for x in model_val]
        df_time = pd.DataFrame(data=data,columns=columns)
        df_time = df_time[df_time["time"]>0.1]

        df_rts = df_rts.merge(df_time, how='left', on=['transp','mach','p_dest'])
        df_rts = df_rts.sort_values(by=['time'])

        fig2, ax2 = plt.subplots()

        for idx, row in df_rts.iterrows():
            ax2.plot([places[row['p_orig']][0], places[row['p_dest']][0]], [places[row['p_orig']][1], places[row['p_dest']][1]], linewidth = 0.5, alpha = 0.3)
            if row['p_dest'] != 'origem':
                ax2.plot(places[row['p_dest']][0], places[row['p_dest']][1],'ro')

        ax2.plot(places['origem'][0], places['origem'][1],'xb')
        plt.show(block = True)





MIP_model = mip_model()
MIP_model.solve_model()
MIP_model.plot_results()
MIP_model.write_output()


