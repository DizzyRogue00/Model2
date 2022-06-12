import gurobipy as gp
import numpy as np
import scipy.stats as st
from gurobipy import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os
import math
import copy
from functools import reduce
import copy
sb.set()

class TwoSize(object):
    def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand,size_type):
        self._beta=0.25
        self._gammar=25
        self._v_w=10
        self._v_v=6
        self._c=16000
        self._e=2400
        self._recovery=0.1359
        self._routeNo=routeNo
        self._distance=distance
        self._average_distance=average_distance
        self._speed=speed
        self._demand=demand
        self._peak_point_demand=peak_point_demand
        self._period=len(speed[0])
        self._size_type=size_type

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self,value):
        self._beta=value

    @property
    def gammar(self):
        return self._gammar
    @gammar.setter
    def gammar(self,value):
        self._gammar=value

    @property
    def v_w(self):
        return self._v_w
    @v_w.setter
    def v_w(self,value):
        self._v_w=value

    @property
    def v_v(self):
        return self._v_v
    @v_v.setter
    def v_v(self,value):
        self._v_v=value

    @property
    def c(self):
        return self._c
    @c.setter
    def c(self,value):
        self._c=value

    @property
    def e(self):
        return self._e
    @e.setter
    def e(self,value):
        self._e=value

    @property
    def recovery_rate(self):
        return self._recovery
    @recovery_rate.setter
    def recovery_rate(self,value):
        self._recovery=value

    def __call__(self, *args, **kwargs):
        result=self.Optimal()
        return result

    def Optimal(self):
        try:
            m=gp.Model('TwoSizeSystemwide')
            m.setParam('nonconvex', 2)

            index_stop_period=gp.tuplelist([(x,y) for x in range(1,self._routeNo+1) for y in range(1,self._period+1)])
            index_stop_period_size=gp.tuplelist([(x, y, z) for x in range(1, self._routeNo + 1) for y in range(1, self._period + 1) for z in range(1,self._size_type+1)])
            index_period_size=gp.tuplelist([(x,y) for x in range(1,self._period+1) for y in range(1,self._size_type+1)])

            size_volume=m.addVars(range(1,self._size_type+1),vtype=GRB.INTEGER,name='size_volume')
            #size_volume = m.addVars(range(1, self._size_type + 1), name='size_volume')
            bus_operating=m.addVars(range(1,self._size_type+1),name='bus_operating')
            h_jts=m.addVars(index_stop_period_size,name='headway_jts')
            h_jts_1=m.addVars(index_stop_period_size,name='headway1_jts')
            h_jts_2=m.addVars(index_stop_period_size,name='headway2_jts')
            h_jts_2_temp = m.addVars(index_stop_period_size, name='headway2_jts_temp')
            delta_jts=m.addVars(index_stop_period_size,vtype=GRB.BINARY,name='delta_jts')
            c_o_j_t_s=m.addVars(index_stop_period_size,name='cost_operator_jts')
            c_uw_j_t_s=m.addVars(index_stop_period_size,name='cost_user_waiting_jts')
            c_uv_j_t_s=m.addVars(index_stop_period_size,name='cost_user_invehicle_jts')
            c_u_j_t_s=m.addVars(index_stop_period_size,name='cost_user_jts')
            c_j_t_s=m.addVars(index_stop_period_size,name='cost_jts')
            c_o_j_t = m.addVars(index_stop_period, name='cost_operator_jt')
            c_uw_j_t=m.addVars(index_stop_period,name='cost_user_waiting_jt')
            c_uv_j_t=m.addVars(index_stop_period,name='cost_user_invehicle_jt')
            c_u_j_t=m.addVars(index_stop_period,name='cost_user_jt')
            c_j_t=m.addVars(index_stop_period,name='cost_jt')
            c_o=m.addVar(name='cost_operator')
            c_uw=m.addVar(name='cost_user_waiting')
            c_uv=m.addVar(name='cost_user_invehicle')
            c_u=m.addVar(name='cost_user')
            c_total=m.addVar(name='cost_total')
            c_p=m.addVar(name='cost_capital')
            n_jts=m.addVars(index_stop_period_size,name='fleet_size_jts')
            n_jt=m.addVars(index_stop_period,name='fleet_size_jt')
            n_ts=m.addVars(index_period_size,name='fleet_size_ts')
            n_s=m.addVars(range(1,self._size_type+1),name='fleet_size_s')
            m.update()

            m.setObjective(c_total+c_p, sense=gp.GRB.MINIMIZE)

            m.addConstr(size_volume[2]-size_volume[1]>=1,name='auxilary_c')
            m.addConstrs((bus_operating[s]==self._gammar+self._beta*size_volume[s] for s in range(1,self._size_type+1)),name='bus_operating_cost')
            m.addConstrs((gp.quicksum(delta_jts[j,t,k] for k in range(1,self._size_type+1))==1 for j,t in index_stop_period),name='assignment_c')
            m.addConstrs((delta_jts[j,t,1]*size_volume[1]+delta_jts[j,t,2]*size_volume[2]>=10**-5 for j,t in index_stop_period),name='assignment_c1')
            m.addConstrs((c_o_j_t_s[j,t,k]*size_volume[k]==2*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*bus_operating[k]/self._speed[j-1][t-1] for j,t,k in index_stop_period_size),name="cost_operator_jts_c")
            m.addConstrs((c_uw_j_t_s[j,t,k]==self.v_w*self._demand[j-1][t-1]*size_volume[k]/self._peak_point_demand[j-1][t-1] for j,t,k in index_stop_period_size),name='cost_user_waiting_jts_c')
            m.addConstrs((c_uv_j_t_s[j,t,k]==self.v_v*2*self._average_distance[j-1]*self._demand[j-1][t-1]/self._speed[j-1][t-1] for j,t,k in index_stop_period_size),name='cost_user_invehicle_jts_c')
            m.addConstrs((c_u_j_t_s[j,t,k]==c_uw_j_t_s[j,t,k]+c_uv_j_t_s[j,t,k] for j,t,k in index_stop_period_size),name='cost_user_jts_c')
            m.addConstrs((c_j_t_s[j,t,k]==c_u_j_t_s[j,t,k]+c_o_j_t_s[j,t,k] for j,t,k in index_stop_period_size),name="cost_jts_c")
            m.addConstrs((c_o_j_t[j,t]==c_o_j_t_s.prod(delta_jts,j,t,'*') for j,t in index_stop_period),name='cost_operator_jt_c')
            m.addConstrs((c_uw_j_t[j,t]==c_uw_j_t_s.prod(delta_jts,j,t,'*') for j,t in index_stop_period),name='cost_user_waiting_jt_c')
            m.addConstrs((c_uv_j_t[j,t]==c_uv_j_t_s.prod(delta_jts,j,t,'*') for j,t in index_stop_period),name='cost_user_invehicle_jt_c')
            m.addConstrs((c_u_j_t[j,t]==c_uw_j_t[j,t]+c_uv_j_t[j,t] for j,t in index_stop_period),name='cost_user_jt_c')
            m.addConstrs((c_j_t[j,t]==c_o_j_t[j,t]+c_u_j_t[j,t] for j,t in index_stop_period),name='cost_jt_c')
            m.addConstr(c_o==gp.quicksum(c_o_j_t[j,t] for j,t in index_stop_period),name='cost_operator_c')
            m.addConstr(c_uw==gp.quicksum(c_uw_j_t[j,t] for j,t in index_stop_period),name='cost_user_waiting_c')
            m.addConstr(c_uv==gp.quicksum(c_uv_j_t[j,t] for j,t in index_stop_period),name='cost_user_invehicle_c')
            m.addConstr(c_u==gp.quicksum(c_u_j_t[j,t] for j,t in index_stop_period),name='cost_user_c')
            m.addConstr(c_total==gp.quicksum(c_j_t[j,t] for j,t in index_stop_period),name='cost_total_c')
            m.addConstrs((h_jts_1[j,t,k]==size_volume[k]/self._peak_point_demand[j-1][t-1] for j,t,k in index_stop_period_size),name='headway1_jts_c')
            m.addConstrs((h_jts_2_temp[j,t,k]==(2*self._distance[j-1]*bus_operating[k]/self._speed[j-1][t-1]/self._demand[j-1][t-1]/self.v_w) for j,t,k in index_stop_period_size),name='headway2_jts_temp_c')
            m.addConstrs((h_jts_2[j,t,k]*h_jts_2[j,t,k]==h_jts_2_temp[j,t,k] for j,t,k in index_stop_period_size),name="headway2_jts_c")
            m.addConstrs((h_jts[j,t,k]==min_(h_jts_1[j,t,k],h_jts_2[j,t,k]) for j,t,k in index_stop_period_size),name='headway_jts_c')
            m.addConstrs((n_jts[j,t,k]*h_jts[j,t,k]==2*self._distance[j-1]/self._speed[j-1][t-1] for j,t,k in index_stop_period_size),name='fleet_size_jts_c')
            m.addConstrs((n_jt[j,t]==n_jts.prod(delta_jts,j,t,'*') for j,t in index_stop_period),name='fleet_size_jt_c')
            m.addConstrs((n_ts[t,k]==n_jts.prod(delta_jts,'*',t,k) for t,k in index_period_size),name='fleet_size_ts_c')
            for k in range(1,self._size_type+1):
                m.addGenConstrMax(n_s[k],[n_ts[t,k] for t in range(1,self._period+1)],name='fleet_size'+str(k)+'_c')
            m.addConstr(c_p==gp.quicksum((self.c+self.e*size_volume[k])*self.recovery_rate/365*n_s[k] for k in range(1,self._size_type+1)),name='cost_capital_c')

            m.optimize()

            if m.status == GRB.OPTIMAL:
                print(m.status)
                self._objVal = m.objVal
                self._result=m.getAttr('x',[c_o,c_uw,c_uv,c_u,c_total,c_p])
                self._size_volume = m.getAttr('x', size_volume)
                self._bus_operating = m.getAttr('x', bus_operating)
                self._h_jts=m.getAttr('x',h_jts)
                self._delta_jts=m.getAttr('x',delta_jts)
                self._n_s=m.getAttr('x',n_s)
                self._h_jts_1=m.getAttr('x',h_jts_1)
                self._h_jts_2=m.getAttr('x',h_jts_2)
                self._c_o_j_t_s=m.getAttr('x',c_o_j_t_s)
                self._c_uw_j_t_s=m.getAttr('x',c_uw_j_t_s)
                self._c_uv_j_t_s=m.getAttr('x',c_uv_j_t_s)
                self._c_u_j_t_s=m.getAttr('x',c_u_j_t_s)
                self._c_j_t_s=m.getAttr('x',c_j_t_s)
                self._c_o_j_t=m.getAttr('x',c_o_j_t)
                self._c_uw_j_t=m.getAttr('x',c_uw_j_t)
                self._c_uv_j_t=m.getAttr('x',c_uv_j_t)
                self._c_u_j_t=m.getAttr('x',c_u_j_t)
                self._c_j_t=m.getAttr('x',c_j_t)
                self._n_jts=m.getAttr('x',n_jts)
                self._n_jt=m.getAttr('x',n_jt)
                self._n_ts=m.getAttr('x',n_ts)
            elif m.status == GRB.TIME_LIMIT:
                m.Params.timeLimit = 200
                if m.MIPGap <= 0.05:
                    print(m.status)
                    print(m.MIPGap)
                    self._objVal = m.objVal
                    self._result = m.getAttr('x', [c_o, c_uw, c_uv, c_u, c_total, c_p])
                    self._size_volume = m.getAttr('x', size_volume)
                    self._bus_operating = m.getAttr('x', bus_operating)
                    self._h_jts = m.getAttr('x', h_jts)
                    self._delta_jts = m.getAttr('x', delta_jts)
                    self._n_s = m.getAttr('x', n_s)
                    self._h_jts_1 = m.getAttr('x', h_jts_1)
                    self._h_jts_2 = m.getAttr('x', h_jts_2)
                    self._c_o_j_t_s = m.getAttr('x', c_o_j_t_s)
                    self._c_uw_j_t_s = m.getAttr('x', c_uw_j_t_s)
                    self._c_uv_j_t_s = m.getAttr('x', c_uv_j_t_s)
                    self._c_u_j_t_s = m.getAttr('x', c_u_j_t_s)
                    self._c_j_t_s = m.getAttr('x', c_j_t_s)
                    self._c_o_j_t = m.getAttr('x', c_o_j_t)
                    self._c_uw_j_t = m.getAttr('x', c_uw_j_t)
                    self._c_uv_j_t = m.getAttr('x', c_uv_j_t)
                    self._c_u_j_t = m.getAttr('x', c_u_j_t)
                    self._c_j_t = m.getAttr('x', c_j_t)
                    self._n_jts = m.getAttr('x', n_jts)
                    self._n_jt = m.getAttr('x', n_jt)
                    self._n_ts = m.getAttr('x', n_ts)
                else:
                    m.Params.MIPGap = 0.05
                    m.optimize()
                    print("OK")
                    print(m.status)
                    self._objVal = m.objVal
                    self._result = m.getAttr('x', [c_o, c_uw, c_uv, c_u, c_total, c_p])
                    self._size_volume = m.getAttr('x', size_volume)
                    self._bus_operating = m.getAttr('x', bus_operating)
                    self._h_jts = m.getAttr('x', h_jts)
                    self._delta_jts = m.getAttr('x', delta_jts)
                    self._n_s = m.getAttr('x', n_s)
                    self._h_jts_1 = m.getAttr('x', h_jts_1)
                    self._h_jts_2 = m.getAttr('x', h_jts_2)
                    self._c_o_j_t_s = m.getAttr('x', c_o_j_t_s)
                    self._c_uw_j_t_s = m.getAttr('x', c_uw_j_t_s)
                    self._c_uv_j_t_s = m.getAttr('x', c_uv_j_t_s)
                    self._c_u_j_t_s = m.getAttr('x', c_u_j_t_s)
                    self._c_j_t_s = m.getAttr('x', c_j_t_s)
                    self._c_o_j_t = m.getAttr('x', c_o_j_t)
                    self._c_uw_j_t = m.getAttr('x', c_uw_j_t)
                    self._c_uv_j_t = m.getAttr('x', c_uv_j_t)
                    self._c_u_j_t = m.getAttr('x', c_u_j_t)
                    self._c_j_t = m.getAttr('x', c_j_t)
                    self._n_jts = m.getAttr('x', n_jts)
                    self._n_jt = m.getAttr('x', n_jt)
                    self._n_ts = m.getAttr('x', n_ts)
            return self._objVal,self._result,self._size_volume,self._bus_operating,self._h_jts,self._delta_jts,self._n_s,self._h_jts_1,self._h_jts_2,self._c_o_j_t_s,self._c_uw_j_t_s,self._c_uv_j_t_s,self._c_u_j_t_s,self._c_j_t_s,self._c_o_j_t,self._c_uw_j_t,self._c_uv_j_t,self._c_u_j_t,self._c_j_t,self._n_jts,self._n_jt,self._n_ts

        except gp.GurobiError as e:
            print('Error code'+str(e.errno)+': '+str(e))
        except AttributeError:
            print('Encountered an attribute error')

    def BFGS(self):
        #delta:{(j,t):(,)}
        #headway:{(j,t):(,)}
        #N_s:(,)
        #s1<s2
        self._maxiter=10
            #10**4
        self._epsilon=10**-5
        self._lambda_init=np.ones(self._size_type)
        self._sigma=1.5
        self._beta1=0.5
        self._mu=2
        self._c1=0.1
        self._c2=0.7
        self._rho=0.55
        self._min_alpha_loss=10**-8
        self._inter_no=10
        self._inter_point_t=1
        self._upper_bound=1000*self._inter_point_t
        self._factor=1.2

        def delta(x):
            x1=x[0]
            x2=x[1]
            def judge(j,t):
                if self._distance[j-1]*self._demand[j-1][t-1]/self._speed[j-1][t-1]<=self._v_w*x1*x2/(2*self._gammar):
                    if x1<=x2:
                        delta_j_t=1,0
                    else:
                        delta_j_t=0,1
                else:
                    if x1<=x2:
                        delta_j_t=0,1
                    else:
                        delta_j_t=1,0
                return delta_j_t
            delta_result={(j,t):judge(j,t) for j in range(1,self._routeNo+1) for t in range(1,self._period+1)}
            return delta_result

        def headway(x,delta_j_t_s):
            #delta_j_t_s=delta(x)
            # def headway_j_t_s(j,t,s):
            #     temp1=[2*self._gammar*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*delta_j_t_s[j,t][s-1]/self._speed[j-1][t-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1)]
            #     temp1=np.sum(temp1)
            #     temp2=[self._v_w*self._demand[j-1][t-1]*delta_j_t_s[j,t][s-1]/self._peak_point_demand[j-1][t-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1)]
            #     temp2=np.sum(temp2)
            #     #print(temp1)
            #     #print(temp2)
            #     #print(np.sqrt(temp1/temp2))
            #     #print(self._peak_point_demand[j-1][t-1])
            #     if temp1==0:
            #         temp_division=0
            #     else:
            #         temp_division=temp1/temp2
            #     headway1_j_t_s=np.sqrt(temp_division)/self._peak_point_demand[j-1][t-1]
            #     headway2_j_t_s=2*self._distance[j-1]*(self._gammar+self._beta*x[s-1])/(self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._v_w)
            #     Headway_j_t_s=np.min([headway1_j_t_s,headway2_j_t_s])
            #     return Headway_j_t_s
            def headway_j_t_s(j,t,s):
                headway1_j_t_s=x[s-1]/self._peak_point_demand[j-1][t-1]
                headway2_j_t_s=np.sqrt(2*self._distance[j-1]*(self._gammar+self._beta*x[s-1])/(self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._v_w))
                Headway_j_t_s=np.min([headway1_j_t_s,headway2_j_t_s])
                return Headway_j_t_s
            headway_result={(j,t):tuple(headway_j_t_s(j,t,s) for s in range(1,self._size_type+1)) for j in range(1,self._routeNo+1) for t in range(1,self._period+1)}
            return headway_result

        def fleet_size(x,delta_jts,headway_jts):
            def N_ts(t,s):
                temp1=[2*self._distance[j-1]*delta_jts[j,t][s-1]/self._speed[j-1][t-1]/headway_jts[j,t][s-1] if headway_jts[j,t][s-1]!=0 else 0 for j in range(1,self._routeNo+1)]
                # #temp1 = [
                #     2 * self._distance[j - 1]  / self._speed[j - 1][t - 1] / headway_jts[j, t][
                #         s - 1] if headway_jts[j, t][s - 1] != 0 else 0 for j in range(1, self._routeNo + 1)]
                temp1=np.sum(temp1)
                return temp1
            N_t=[tuple(N_ts(t,s) for s in range(1,self._size_type+1)) for t in range(1,self._period+1)]
            N_t_temp=[[N_t[j][i] for j in range(len(N_t))] for i in range(self._size_type)]
            N_s=np.max(N_t_temp,axis=1)
            N_s=tuple(N_s)
            return N_s

        def ctp(x,delta_jts,headway_jts,fleet_size_s):
            c_o_temp=[(2*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*(self._gammar+self._beta*x[s-1])/self._speed[j-1][t-1]/x[s-1])*delta_jts[j,t][s-1] if x[s-1]!=0 else 0 for j in range(1,self._routeNo+1) for t in range(1,self._period+1) for s in range(1,self._size_type+1)]
            c_uw_temp=[self._v_w*self._demand[j-1][t-1]*x[s-1]/self._peak_point_demand[j-1][t-1]*delta_jts[j,t][s-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1) for s in range(1,self._size_type+1)]
            # c_uw_temp = [self._v_w * self._demand[j - 1][t - 1] * headway_jts[j,t][s-1] *
            #              delta_jts[j, t][s - 1] for j in range(1, self._routeNo + 1) for t in range(1, self._period + 1)
            #              for s in range(1, self._size_type + 1)]
            c_uv_temp=[(2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1])*delta_jts[j,t][s-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1) for s in range(1,self._size_type+1)]
            c_o=np.sum(c_o_temp)
            c_uw=np.sum(c_uw_temp)
            c_uv=np.sum(c_uv_temp)
            c_u=c_uw+c_uv
            c_total=c_o+c_u
            c_p_temp=[((self._c+self._e*x[s-1])*self._recovery/365)*fleet_size_s[s-1] for s in range(1,self._size_type+1)]
            c_p=np.sum(c_p_temp)
            c_tp=c_total+c_p
            #c_tp_a=c_tp+1/(2*self._mu)*np.sum([np.power(np.max([0,self._lambda_init[s-1]-self._mu*(x[s-1]-self._epsilon)]),2)-np.power(self._lambda_init[s-1],2) for s in range(1,self._size_type+1)])
            #c_tp_inter_point=c_tp-1/self._inter_point_t*np.sum([np.log(x[s-1]-self._epsilon) if x[s-1]-self._epsilon>0 else -100000 for s in range(1,self._size_type+1)])
            c_tp_inter_point = c_tp - 1 / self._inter_point_t * np.sum(
                [np.log(x[s - 1] - 1) for s in
                 range(1, self._size_type + 1)])
            # if x[1]-x[0]-self._epsilon>=0:
            #     temp=np.log(x[1]-x[0]-self._epsilon)
            # else:
            #     temp=-100000
            # c_tp_inter_point=c_tp_inter_point-1/self._inter_point_t*temp
            return c_tp_inter_point,c_tp,c_o,c_uw,c_uv,c_u,c_total,c_p

        def num_grad(x,delta_jts,headway_jts,fleet_size_s):
            x=list(x)
            def num_grad_item(h,i):
                x1=copy.copy(x)
                x2=copy.copy(x)
                x1[i]=x1[i]-h
                x2[i]=x2[i]+h
                y1,y2=ctp(x1,delta_jts,headway_jts,fleet_size_s)[0],ctp(x2,delta_jts,headway_jts,fleet_size_s)[0]
                df=(y2-y1)/(2*h)
                return df
            df=[num_grad_item(10**-6,i) for i in range(len(x))]
            df=np.array(df)
            return df

        def num_hess(x,delta_jts,headway_jts,fleet_size_s):
            x=list(x)
            def hess_row(h,i):
                x1=copy.copy(x)
                x1[i]=x1[i]-h
                df1=num_grad(x1,delta_jts,headway_jts,fleet_size_s)
                # #df1=df1-np.sum(
                #  [np.max([self._lambda_init[s - 1] - self._mu * (x1[s - 1]-self._epsilon), 0]) for s in
                #   range(1, self._size_type + 1)])
                x2=copy.copy(x)
                x2[i]=x2[i]+h
                df2=num_grad(x2,delta_jts,headway_jts,fleet_size_s)
                # df2=df2-np.sum(
                #  [np.max([self._lambda_init[s - 1] - self._mu * (x2[s - 1]-self._epsilon), 0]) for s in
                #   range(1, self._size_type + 1)])
                d2f=(df2-df1)/(2*h)
                return d2f
            hess=[hess_row(10**-6,i) for i in range(len(x))]
            hess=np.array(hess)
            return hess

        def linesearch(x,dk,delta_jts,headway_jts,fleet_size_s):
            a_min=0
            a_max=10000
            x=np.array(x)
            ak=(a_min+a_max)/2
            grad_x = num_grad(x, delta_jts, headway_jts,fleet_size_s)
            # grad_x = grad_x - np.sum(
            #      [np.max([self._lambda_init[s - 1] - self._mu * (x[s - 1]-self._epsilon), 0]) for s in
            #       range(1, self._size_type + 1)])
            phi_x = ctp(x, delta_jts, headway_jts,fleet_size_s)[0]
            phi_grad_x = np.dot(dk.reshape(1, dk.shape[0]), grad_x)
            iter_ak=0
            iter_max=100
            while True:
                if iter_ak>=iter_max:
                    return ak
                x_a=x+ak*dk
                # delta_a = delta(x_a)
                # headway_a = headway(x_a, delta_a)
                # fleet_size_a = fleet_size(x_a, delta_a, headway_a)
                grad_a = num_grad(x_a, delta_jts, headway_jts,fleet_size_s)
                #grad_a = num_grad(x_a, delta_a, headway_a, fleet_size_a)
                # grad_a = grad_a - np.sum(
                #     [np.max([self._lambda_init[s - 1] - self._mu * (x_a[s - 1]-self._epsilon), 0]) for s in
                #      range(1, self._size_type + 1)])
                phi_a = ctp(x_a, delta_jts, headway_jts,fleet_size_s)[0]
                phi_grad_a = np.dot(dk.reshape(1, dk.shape[0]), grad_a)
                # phi_a = ctp(x_a, delta_a, headway_a, fleet_size_a)[0]
                # phi_grad_a = np.dot(dk.reshape(1, dk.shape[0]), grad_a)
                if phi_a<=phi_x+self._c1*ak*phi_grad_x:
                    if np.abs(phi_grad_a)<=self._c2*np.abs(phi_grad_x):
                        return ak
                    else:
                        a_min=ak
                        if a_max<10000:
                            ak=(a_min+a_max)/2
                        else:
                            ak*=2
                else:
                    a_max=ak
                    ak=(a_min+a_max)/2
                iter_ak+=1
            # def zoom(a_low,a_high):
            #     grad_x=num_grad(x,delta_jts,fleet_size_s)
            #     grad_x= grad_x - np.sum(
            #         [np.max([self._lambda_init[s - 1] - self._mu * x[s - 1], 0]) for s in
            #          range(1, self._size_type + 1)])
            #     phi_x=ctp(x,delta_jts,fleet_size_s)[0]
            #     phi_grad_x=np.dot(dk.reshape(1,dk.shape[0]),grad_x)
            #     while True:
            #         if abs(a_low-a_high)<=self._min_alpha_loss:
            #             alpha=a_low
            #             return alpha
            #         x_a_low=x+a_low*dk
            #         x_a_high=x+a_high*dk
            #         phi_a_low=ctp(x_a_low,delta_jts,fleet_size_s)[0]
            #         phi_a_high = ctp(x_a_high, delta_jts, fleet_size_s)[0]
            #         grad_x_a_low=num_grad(x_a_low,delta_jts,fleet_size_s)
            #         grad_x_a_low=grad_x_a_low-np.sum([np.max([self._lambda_init[s - 1] - self._mu * x_a_low[s - 1], 0]) for s in range(1, self._size_type + 1)])
            #         phi_grad_x_a_low=np.dot(dk.reshape(1,dk.shape[0]),grad_x_a_low)
            #         # print('phi_a_low:',phi_a_low)
            #         #print('phi_a_high:',phi_a_high)
            #         print(phi_grad_x_a_low)
            #         print('a_low:',a_low)
            #         print('a_high:',a_high)
            #         alpha=a_low-(a_low-a_high)/(2*(1-(phi_a_low-phi_a_high)/((a_low-a_high)*phi_grad_x_a_low)))
            #         #print('alpha:',alpha)
            #         x_alpha=x+alpha*dk
            #         phi_alpha=ctp(x_alpha,delta_jts,fleet_size_s)[0]
            #         grad_alpha = num_grad(x_alpha, delta_jts, fleet_size_s)
            #         grad_alpha = grad_alpha - np.sum(
            #             [np.max([self._lambda_init[s - 1] - self._mu * x_alpha[s - 1], 0]) for s in
            #              range(1, self._size_type + 1)])
            #         phi_grad_alpha = np.dot(dk.reshape(1, dk.shape[0]), grad_alpha)
            #         # print('phi_grad_alpha:',phi_grad_alpha)
            #         # print('second:',-self._c2*phi_grad_x)
            #         if phi_alpha>phi_x+self._c1*alpha*phi_grad_x or phi_alpha>=phi_a_low:
            #             a_high=copy.copy(alpha)
            #         else:
            #             if np.abs(phi_grad_alpha)<=np.abs(self._c2*phi_grad_x):
            #                 a_star=alpha
            #                 return a_star
            #             if phi_grad_alpha*(a_high-a_low)>=0:
            #                 a_high=copy.copy(a_low)
            #             a_low=copy.copy(alpha)
            #
            # x=np.array(x)
            # grad_x = num_grad(x, delta_jts, fleet_size_s)
            # grad_x = grad_x - np.sum(
            #     [np.max([self._lambda_init[s - 1] - self._mu * x[s - 1], 0]) for s in
            #      range(1, self._size_type + 1)])
            # phi_x = ctp(x, delta_jts, fleet_size_s)[0]
            # phi_grad_x = np.dot(dk.reshape(1, dk.shape[0]), grad_x)
            #
            # a_i_1=0
            # x_a_i_1=x+a_i_1*dk
            # phi_a_i_1=ctp(x_a_i_1,delta_jts,fleet_size_s)
            # a_max=10000
            # a_i=a_i_1+(a_max-a_i_1)*np.random.random()
            # while True:
            #     print('ak:',a_i)
            #     x_a=x+a_i*dk
            #     phi_a=ctp(x_a,delta_jts,fleet_size_s)[0]
            #     grad_a = num_grad(x_a, delta_jts, fleet_size_s)
            #     grad_a = grad_a - np.sum(
            #         [np.max([self._lambda_init[s - 1] - self._mu * x_a[s - 1], 0]) for s in
            #          range(1, self._size_type + 1)])
            #     phi_grad_a = np.dot(dk.reshape(1, dk.shape[0]), grad_a)
            #     print('a_i_1,',a_i_1)
            #     print('a_i,',a_i)
            #     if phi_a>phi_x+self._c1*a_i*phi_grad_x or phi_a>=phi_a_i_1:
            #         print('OK1')
            #         ak=zoom(a_i_1,a_i)
            #
            #         return ak
            #     if np.abs(phi_grad_a)<=-self._c2*phi_grad_x:
            #         print('OK2')
            #         ak=a_i
            #
            #         return ak
            #     if phi_grad_a>=0:
            #         print('OK3')
            #         ak=zoom(a_i,a_i_1)
            #
            #         return ak
            #     a_i_1=copy.copy(a_i)
            #     x_a_i_1 = x + a_i_1 * dk
            #     phi_a_i_1 = ctp(x_a_i_1, delta_jts, fleet_size_s)
            #     a_i=a_i+(a_max-a_i)*np.random.random()

            # x=np.array(x)
            # grad_x=num_grad(x,delta_jts,fleet_size_s)
            # grad_x = grad_x - np.sum([np.max([self._lambda_init[s - 1] - self._mu * x[s - 1], 0]) for s in range(1, self._size_type + 1)])
            # ak=1
            # x1=x+ak*dk
            # #delta1=delta(x1)
            # #headway1=headway(x1,delta1)
            # #fleet_size1=fleet_size(x1,delta1,headway1)
            # #grad_x1=num_grad(x1,delta1,fleet_size1)
            # #grad_x1 = grad_x1 - np.sum([np.max([self._lambda_init[s - 1] - self._mu * x1[s - 1], 0]) for s in range(1, self._size_type + 1)])
            # grad_x1 = num_grad(x1, delta_jts, fleet_size_s)
            # grad_x1 = grad_x1 - np.sum(
            #     [np.max([self._lambda_init[s - 1] - self._mu * x1[s - 1], 0]) for s in range(1, self._size_type + 1)])
            # if not (ctp(x1,delta_jts,fleet_size_s)[0]<=ctp(x,delta_jts,fleet_size_s)[0]+self._c1*ak*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk) and \
            #     np.abs(np.dot(grad_x1.reshape(1,grad_x1.shape[0]),dk))<=-self._c2*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk)):
            #     ak=1
            #     while ctp(x1,delta_jts,fleet_size_s)[0]>ctp(x,delta_jts,fleet_size_s)[0]+self._c1*ak*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk):
            #         ak*=self._rho
            #         print('1:',ak)
            #         x1 = x + ak * dk
            #         #delta1 = delta(x1)
            #         #headway1 = headway(x1, delta1)
            #         #fleet_size1 = fleet_size(x1, delta1, headway1)
            #         grad_x1 = num_grad(x1, delta_jts, fleet_size_s)
            #         grad_x1 = grad_x1 - np.sum([np.max([self._lambda_init[s - 1] - self._mu * x1[s - 1], 0]) for s in range(1, self._size_type + 1)])
            #     while np.abs(np.dot(grad_x1.reshape(1,grad_x1.shape[0]),dk))>-self._c2*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk):
            #         a1=ak/self._rho
            #         print('2:',ak)
            #         da=a1-ak
            #         print(da)
            #         x1 = x + (ak+da) * dk
            #         #delta1 = delta(x1)
            #         #headway1 = headway(x1, delta1)
            #         #fleet_size1 = fleet_size(x1, delta1, headway1)
            #         while ctp(x1, delta_jts, fleet_size_s)[0] > ctp(x, delta_jts, fleet_size_s)[0] + self._c1 * ak * np.dot(
            #             grad_x.reshape(1, grad_x.shape[0]), dk):
            #             da*=self._rho
            #             print('da:',da)
            #             x1 = x + (ak + da) * dk
            #             #delta1 = delta(x1)
            #             #headway1 = headway(x1, delta1)
            #             #fleet_size1 = fleet_size(x1, delta1, headway1)
            #         grad_x1 = num_grad(x1, delta_jts, fleet_size_s)
            #         grad_x1 = grad_x1 - np.sum([np.max([self._lambda_init[s - 1] - self._mu * x1[s - 1], 0]) for s in
            #                               range(1, self._size_type + 1)])
            #         ak+=da
            # if not (ctp(x1,delta1,fleet_size1)[0]<=ctp(x,delta_jts,fleet_size_s)[0]+self._c1*ak*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk) and \
            #     np.abs(np.dot(grad_x1.reshape(1,grad_x1.shape[0]),dk))<=-self._c2*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk)):
            #     ak=1
            #     while ctp(x1,delta1,fleet_size1)[0]>ctp(x,delta_jts,fleet_size_s)[0]+self._c1*ak*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk):
            #         ak*=self._rho
            #         print('1:',ak)
            #         x1 = x + ak * dk
            #         delta1 = delta(x1)
            #         headway1 = headway(x1, delta1)
            #         fleet_size1 = fleet_size(x1, delta1, headway1)
            #         grad_x1 = num_grad(x1, delta1, fleet_size1)
            #         grad_x1 = grad_x1 - np.sum([np.max([self._lambda_init[s - 1] - self._mu * x1[s - 1], 0]) for s in range(1, self._size_type + 1)])
            #     while np.abs(np.dot(grad_x1.reshape(1,grad_x1.shape[0]),dk))>-self._c2*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk):
            #         print(np.abs(np.dot(grad_x1.reshape(1,grad_x1.shape[0]),dk)))
            #         print(-self._c2*np.dot(grad_x.reshape(1, grad_x.shape[0]), dk))
            #         a1=ak/self._rho
            #         print('2:',ak)
            #         da=a1-ak
            #         print(da)
            #         x1 = x + (ak+da) * dk
            #         delta1 = delta(x1)
            #         headway1 = headway(x1, delta1)
            #         fleet_size1 = fleet_size(x1, delta1, headway1)
            #         print(ctp(x1, delta1, fleet_size1)[0])
            #         print(ctp(x, delta_jts, fleet_size_s)[0] + self._c1 * ak * np.dot(
            #             grad_x.reshape(1, grad_x.shape[0]), dk))
            #         print(ctp(x1, delta1, fleet_size1)[0] > ctp(x, delta_jts, fleet_size_s)[0] + self._c1 * ak * np.dot(
            #             grad_x.reshape(1, grad_x.shape[0]), dk))
            #         while ctp(x1, delta1, fleet_size1)[0] > ctp(x, delta_jts, fleet_size_s)[0] + self._c1 * ak * np.dot(
            #             grad_x.reshape(1, grad_x.shape[0]), dk):
            #             da*=self._rho
            #             print('da:',da)
            #             x1 = x + (ak + da) * dk
            #             delta1 = delta(x1)
            #             headway1 = headway(x1, delta1)
            #             fleet_size1 = fleet_size(x1, delta1, headway1)
            #         grad_x1 = num_grad(x1, delta1, fleet_size1)
            #         grad_x1 = grad_x1 - np.sum([np.max([self._lambda_init[s - 1] - self._mu * x1[s - 1], 0]) for s in
            #                               range(1, self._size_type + 1)])
            #         ak+=da
            #return ak
            # for i in range(20):
            #     newf,oldf=ctp(x+ak*dk,delta,fleet_size)[0],ctp(x,delta,fleet_size)[0]
            #     if newf<oldf:
            #         return ak
            #     else:
            #         ak=ak/4
            # return ak

        #initial solution
        volume_min = np.min(self._peak_point_demand)
        volume_max=np.max(self._peak_point_demand)
        s_min_temp=[2*self._gammar*self._distance[j-1]*volume_min/self._speed[j-1][t-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1)]
        s_min=np.sqrt(np.sum(s_min_temp)/(self._v_w*self._routeNo*self._period))
        s_max_temp = [2 * self._gammar * self._distance[j - 1] * volume_max / self._speed[j - 1][t - 1] for j in
                      range(1, self._routeNo + 1) for t in range(1, self._period + 1)]
        s_max = np.sqrt(np.sum(s_max_temp) / (self._v_w * self._routeNo * self._period))
        x_init=np.array([s_min,s_max])
        #delta_init=delta(x_init)
        #headway_init=headway(x_init,delta_init)
        #fleet_size_init=fleet_size(x_init,delta_init,headway_init)
        #ctp_init=ctp(x_init,delta_init,fleet_size_init)
        def BFGS_item(x):
            # iter_no=0
            # delta_x=delta(x)
            # headway_x=headway(x,delta_x)
            # fleet_size_x=fleet_size(x,delta_x,headway_x)
            # ctp_x=ctp(x,delta_x,fleet_size_x)
            # grad=num_grad(x,delta_x,fleet_size_x)
            # grad=grad-np.sum([np.max([self._lambda_init[s - 1] - self._mu * x[s - 1], 0]) for s in range(1, self._size_type + 1)])
            # #Bk=np.eye(x.size)
            # while True:
            #     print(iter_no, {"solution": x, "objValue": ctp_x})
            #     yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,"objValue": ctp_x}
            #     tar = np.sqrt(np.sum([np.power(np.min([x[s - 1], 0]), 2) for s in range(1, self._size_type + 1)]))
            #     Hk = np.eye(x.size)
            #     inter_iter=1
            #     while True:
            #         print(np.linalg.norm(grad))
            #         dk=-np.dot(Hk,grad)
            #         ak=linesearch(x,dk,delta_x,fleet_size_x)
            #         print(ak)
            #         x=x+ak*dk
            #         delta_x = delta(x)
            #         headway_x = headway(x, delta_x)
            #         fleet_size_x = fleet_size(x, delta_x, headway_x)
            #         ctp_x = ctp(x, delta_x, fleet_size_x)
            #         grad_x=num_grad(x,delta_x,fleet_size_x)
            #         grad_x=grad_x-np.sum([np.max([self._lambda_init[s - 1] - self._mu * x[s - 1], 0]) for s in range(1, self._size_type + 1)])
            #         if np.linalg.norm(grad_x)<self._epsilon:
            #             break
            #         if inter_iter==self._size_type:
            #             Hk=np.eye(x.size)
            #             inter_iter=1
            #             continue
            #         #BFGS
            #         sk=ak*dk
            #         yk=grad_x-grad
            #         grad = copy.copy(grad_x)
            #         Hk=Hk+(1+np.dot(np.dot(yk.reshape(1,yk.shape[0]),Hk),yk)/np.dot(sk.reshape(1,sk.shape[0]),yk))*(np.dot(sk.reshape(sk.shape[0],1),sk.reshape(1,sk.shape[0]))/np.dot(sk.reshape(1,sk.shape[0]),yk))-(np.dot(np.dot(sk.reshape(sk.shape[0],1),yk.reshape(1,yk.shape[0])),Hk)+np.dot(np.dot(Hk,yk).reshape(Hk.shape[0],1),sk.reshape(1,sk.shape[0])))/np.dot(sk.reshape(1,sk.shape[0]),yk)
            #         print(x)
            #         print(sk)
            #         print(yk)
            #         print(ctp_x[0])
            #         print(ctp_x[1])
            #         inter_iter+=1
            #         #Bk=Bk-np.dot(np.dot(np.dot(Bk,sk).reshape(sk.shape[0],1),sk.reshape(1,sk.shape[0])),Bk)/np.dot(np.dot(sk.reshape(1,sk.shape[0]),Bk),sk)+np.dot(yk.reshape(yk.shape[0],1),yk.reshape(1,yk.shape[0]))/np.dot(yk.reshape(1,yk.shape[0]),sk)
            #         #if np.dot(yk.reshape(1, grad.shape[0]), sk) > 0:
            #         #    Bk = Bk - np.dot(np.dot(np.dot(Bk, sk).reshape(sk.shape[0], 1), sk.reshape(1, sk.shape[0])),Bk) / np.dot(np.dot(sk.reshape(1, sk.shape[0]), Bk), sk) + np.dot(yk.reshape(yk.shape[0], 1), yk.reshape(1, yk.shape[0])) / np.dot(yk.reshape(1, yk.shape[0]), sk)
            #     tar1 = np.sqrt(np.sum([np.power(np.min([x[s - 1], 0]), 2) for s in range(1, self._size_type + 1)]))
            #     if tar1<=self._epsilon:
            #         yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,"objValue": ctp_x}
            #         return
            #     if tar==0:
            #         self._mu=self._mu
            #     elif tar1/tar>=self._beta1:
            #         self._mu*=self._sigma
            #     print(x)
            #     self._lambda_init=np.array([np.max([self._lambda_init[s-1]-self._mu*x[s-1],0]) for s in range(1,self._size_type+1)])
            #     print(self._lambda_init)
            #     if iter_no>self._maxiter:
            #         yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
            #                         "objValue": ctp_x}
            #         return
            #     iter_no+=1
            iter_no = 0
            delta_x = delta(x)
            headway_x = headway(x, delta_x)
            fleet_size_x = fleet_size(x, delta_x, headway_x)
            ctp_x = ctp(x, delta_x,headway_x, fleet_size_x)
            grad = num_grad(x, delta_x, headway_x,fleet_size_x)
            grad = grad - np.sum(
                [np.max([self._lambda_init[s - 1] - self._mu * (x[s - 1]-self._epsilon), 0]) for s in range(1, self._size_type + 1)])
            # Bk=np.eye(x.size)
            while True:
                print(iter_no, {"solution": x, "objValue": ctp_x})
                print(delta_x)
                print(headway_x)
                print(fleet_size_x)
                yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                "objValue": ctp_x}
                tar = np.sqrt(np.sum([np.power(np.min([x[s - 1]-self._epsilon, 0]), 2) for s in range(1, self._size_type + 1)]))
                Hk = np.eye(x.size)
                inter_iter = 1
                while True:
                    grad = num_grad(x, delta_x, headway_x,fleet_size_x)
                    grad = grad - np.sum(
                        [np.max([self._lambda_init[s - 1] - self._mu * (x[s - 1]-self._epsilon), 0]) for s in
                         range(1, self._size_type + 1)])
                    print('norm_grad:',np.linalg.norm(grad))
                    if np.linalg.norm(grad) < self._epsilon:
                        break
                    dk = -np.dot(Hk, grad)
                    print('dk_original:',dk)
                    print('able to descent',np.dot(dk.reshape(1,dk.shape[0]),grad))
                    dk=dk/np.linalg.norm(dk)
                    print('dk_normal:',dk)
                    ak = linesearch(x, dk, delta_x,headway_x, fleet_size_x)
                    print('ak_BFGS:',ak)
                    x_new = x + ak * dk
                    print('x_old:',x)
                    print('x_new:',x_new)
                    #x_new=np.array([x_new[i] if x_new[i]>0 else self._epsilon for i in range(self._size_type)])
                    #print('x_new_modify:',x_new)
                    print('ak*dk',ak*dk)
                    #delta_x = delta(x)
                    #headway_x = headway(x, delta_x)
                    #fleet_size_x = fleet_size(x, delta_x, headway_x)
                    #ctp_x = ctp(x, delta_x, fleet_size_x)
                    grad_x = num_grad(x_new, delta_x,headway_x, fleet_size_x)
                    grad_x = grad_x - np.sum([np.max([self._lambda_init[s - 1] - self._mu * (x_new[s - 1]-self._epsilon), 0]) for s in
                                              range(1, self._size_type + 1)])

                    # if inter_iter == self._size_type:
                    #     Hk = np.eye(x.size)
                    #     grad=copy.copy(grad_x)
                    #     inter_iter = 1
                    #     continue
                    # BFGS
                    sk = x_new-x
                    if np.linalg.norm(sk)<=self._epsilon:
                        break
                    #sk=sk/np.linalg.norm(sk)
                    yk = grad_x - grad
                    #yk=yk/np.linalg.norm(yk)
                    Hk = Hk + (1 + np.dot(np.dot(yk.reshape(1, yk.shape[0]), Hk), yk) / np.dot(
                        sk.reshape(1, sk.shape[0]), yk)) * (
                                     np.dot(sk.reshape(sk.shape[0], 1), sk.reshape(1, sk.shape[0])) / np.dot(
                                 sk.reshape(1, sk.shape[0]), yk)) - (
                                     np.dot(np.dot(sk.reshape(sk.shape[0], 1), yk.reshape(1, yk.shape[0])),
                                            Hk) + np.dot(np.dot(Hk, yk).reshape(Hk.shape[0], 1),
                                                         sk.reshape(1, sk.shape[0]))) / np.dot(
                        sk.reshape(1, sk.shape[0]), yk)
                    print('sk:',sk)
                    print('yk:',yk)
                    print('Hk:',Hk)
                    print('Obj:',ctp(x_new,delta_x,headway_x,fleet_size_x)[0])
                    x=copy.copy(x_new)
                    #print(ctp_x[1])
                    inter_iter += 1
                    # Bk=Bk-np.dot(np.dot(np.dot(Bk,sk).reshape(sk.shape[0],1),sk.reshape(1,sk.shape[0])),Bk)/np.dot(np.dot(sk.reshape(1,sk.shape[0]),Bk),sk)+np.dot(yk.reshape(yk.shape[0],1),yk.reshape(1,yk.shape[0]))/np.dot(yk.reshape(1,yk.shape[0]),sk)
                    # if np.dot(yk.reshape(1, grad.shape[0]), sk) > 0:
                    #    Bk = Bk - np.dot(np.dot(np.dot(Bk, sk).reshape(sk.shape[0], 1), sk.reshape(1, sk.shape[0])),Bk) / np.dot(np.dot(sk.reshape(1, sk.shape[0]), Bk), sk) + np.dot(yk.reshape(yk.shape[0], 1), yk.reshape(1, yk.shape[0])) / np.dot(yk.reshape(1, yk.shape[0]), sk)
                delta_x = delta(x)
                headway_x = headway(x, delta_x)
                fleet_size_x = fleet_size(x, delta_x, headway_x)
                ctp_x = ctp(x, delta_x,headway_x, fleet_size_x)
                tar1 = np.sqrt(np.sum([np.power(np.min([x[s - 1]-self._epsilon, 0]), 2) for s in range(1, self._size_type + 1)]))
                if tar1 <= self._epsilon:
                    yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                    "objValue": ctp_x}
                    return
                if tar == 0:
                    self._mu = self._mu
                elif tar1 / tar >= self._beta1:
                    self._mu *= self._sigma
                print(x)
                self._lambda_init = np.array([np.max([self._lambda_init[s - 1] - self._mu * (x[s - 1]-self._epsilon), 0]) for s in
                                              range(1, self._size_type + 1)])
                print('lambda:',self._lambda_init)
                print('mu:',self._mu)
                if iter_no > self._maxiter:
                    yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                    "objValue": ctp_x}
                    return
                iter_no += 1
        def steepest_item(x):
            iter_no = 0
            delta_x = delta(x)
            headway_x = headway(x, delta_x)
            fleet_size_x = fleet_size(x, delta_x, headway_x)
            ctp_x = ctp(x, delta_x,headway_x, fleet_size_x)
            grad = num_grad(x, delta_x, headway_x,fleet_size_x)
            while True:
                x_initial=copy.copy(x)
                inter_iter = 1
                if self._inter_point_t>self._upper_bound:
                    return
                while True:
                    print(iter_no,self._inter_point_t, {"solution": x, "objValue": ctp_x})
                    print(delta_x)
                    print(headway_x)
                    print(fleet_size_x)
                    yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                    "objValue": ctp_x}
                    if inter_iter>self._inter_no:
                        break
                    grad = num_grad(x, delta_x, headway_x,fleet_size_x)
                    # grad = grad - np.sum(
                    #     [np.max([self._lambda_init[s - 1] - self._mu * (x[s - 1]-self._epsilon), 0]) for s in
                    #      range(1, self._size_type + 1)])
                    print('norm_grad:',np.linalg.norm(grad))
                    if np.linalg.norm(grad) < self._epsilon:
                        break
                    dk = -grad
                    print('dk_original:',dk)
                    #dk=dk/np.linalg.norm(dk)
                    #print('dk_normal:',dk)
                    ak = linesearch(x, dk, delta_x,headway_x, fleet_size_x)
                    #ak=1
                    print('ak_Steepest:',ak)
                    x_new = x + ak * dk
                    #x_new=np.array([x_new[i] if x_new[i]>0 else self._epsilon for i in range(self._size_type)])
                    delta_x = delta(x_new)
                    headway_x = headway(x_new, delta_x)
                    fleet_size_x = fleet_size(x_new, delta_x, headway_x)
                    ctp_x=ctp(x_new,delta_x,headway_x,fleet_size_x)
                    if np.linalg.norm(x_new-x)<=self._epsilon:
                        print(iter_no, self._inter_point_t, {"solution": x_new, "objValue": ctp_x})
                        print(delta_x)
                        print(headway_x)
                        print(fleet_size_x)
                        x = copy.copy(x_new)
                        yield iter_no,{"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                        "objValue": ctp_x}
                        break
                    print('x_old:',x)
                    print('x_new:',x_new)
                    print('Obj:',ctp(x_new,delta_x,headway_x,fleet_size_x)[0])
                    x=copy.copy(x_new)
                    inter_iter += 1
                    iter_no+=1
                delta_x = delta(x)
                headway_x = headway(x, delta_x)
                fleet_size_x = fleet_size(x, delta_x, headway_x)
                ctp_x = ctp(x, delta_x,headway_x, fleet_size_x)
                if iter_no>self._maxiter:
                    yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x,
                                                         "fleet": fleet_size_x,
                                                         "objValue": ctp_x}
                diff=x-x_initial
                error=np.linalg.norm(diff)
                self._inter_point_t*=self._factor
                print('t:',self._inter_point_t)
                if error<=self._epsilon:
                    return

        def newton_item(x):
            iter_no = 0
            delta_x = delta(x)
            headway_x = headway(x, delta_x)
            fleet_size_x = fleet_size(x, delta_x, headway_x)
            ctp_x = ctp(x, delta_x, headway_x, fleet_size_x)
            grad = num_grad(x, delta_x, headway_x, fleet_size_x)
            while True:
                x_initial = copy.copy(x)
                inter_iter = 1
                if self._inter_point_t > self._upper_bound:
                    return
                while True:
                    print(iter_no, self._inter_point_t, {"solution": x, "objValue": ctp_x})
                    print(delta_x)
                    print(headway_x)
                    print(fleet_size_x)
                    yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                    "objValue": ctp_x}
                    if inter_iter > self._inter_no:
                        break
                    grad = num_grad(x, delta_x, headway_x, fleet_size_x)
                    print('norm_grad:', np.linalg.norm(grad))
                    if np.linalg.norm(grad) < self._epsilon:
                        break
                    hess=num_hess(x,delta_x,headway_x,fleet_size_x)
                    dk = -np.dot(np.linalg.inv(hess),grad)
                    print('dk_original:', dk)
                    # dk=dk/np.linalg.norm(dk)
                    # print('dk_normal:',dk)
                    ak = linesearch(x, dk, delta_x, headway_x, fleet_size_x)
                    print('ak_Newton:', ak)
                    x_new = x + ak * dk
                    # x_new=np.array([x_new[i] if x_new[i]>0 else self._epsilon for i in range(self._size_type)])
                    delta_x = delta(x_new)
                    headway_x = headway(x_new, delta_x)
                    fleet_size_x = fleet_size(x_new, delta_x, headway_x)
                    ctp_x = ctp(x_new, delta_x, headway_x, fleet_size_x)
                    if np.linalg.norm(x_new - x) <= self._epsilon:
                        print(iter_no, self._inter_point_t, {"solution": x_new, "objValue": ctp_x})
                        print(delta_x)
                        print(headway_x)
                        print(fleet_size_x)
                        x = copy.copy(x_new)
                        yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x, "fleet": fleet_size_x,
                                        "objValue": ctp_x}
                        break
                    print('x_old:', x)
                    print('x_new:', x_new)
                    print('Obj:', ctp(x_new, delta_x, headway_x, fleet_size_x)[0])
                    x = copy.copy(x_new)
                    inter_iter += 1
                    iter_no += 1
                delta_x = delta(x)
                headway_x = headway(x, delta_x)
                fleet_size_x = fleet_size(x, delta_x, headway_x)
                ctp_x = ctp(x, delta_x, headway_x, fleet_size_x)
                if iter_no > self._maxiter:
                    yield iter_no, {"solution": x, "delta": delta_x, "headway": headway_x,
                                    "fleet": fleet_size_x,
                                    "objValue": ctp_x}
                diff = x - x_initial
                error = np.linalg.norm(diff)
                self._inter_point_t *= self._factor
                print('t:', self._inter_point_t)
                if error <= self._epsilon:
                    return

        #result=BFGS_item(x_init)
        result=steepest_item(x_init)
        #result = steepest_item(np.array([100,2]))
        #result = steepest_item(np.array([19,30]))
        #result=newton_item(x_init)
        #result = BFGS_item(np.array([100,85]))
        result=dict(result)
        return result


