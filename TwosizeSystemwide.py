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

            size_volume=m.addVars(range(1,self._size_type+1),name='size_volume')
            bus_operating=m.addVars(range(1,self._size_type+1),name='bus_operating')
            h_jts=m.addVars(index_stop_period_size,name='headway_jts')
            h_jts_1=m.addVars(index_stop_period_size,name='headway1_jts')
            h_jts_2=m.addVars(index_stop_period_size,name='headway2_jts')
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

            m.addConstrs((bus_operating[s]==self._gammar+self._beta*size_volume[s] for s in range(1,self._size_type+1)),name='bus_operating_cost')
            m.addConstrs((gp.quicksum(delta_jts[j,t,k] for k in range(1,self._size_type+1))==1 for j,t in index_stop_period),name='assignment_c')
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
            m.addConstrs((h_jts_2[j,t,k]==2*self._distance[j-1]*bus_operating[k]/self._speed[j-1][t-1]/self._demand[j-1][t-1]/self.v_w for j,t,k in index_stop_period_size),name='headway2_jts_c')
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
        def delta(x):
            x1=x[0]
            x2=x[1]
            def judge(j,t):
                if self._distance[j-1]*self._demand[j-1][t-1]/self._speed[j-1][t-1]<=self._v_w*x1*x2/(2*self._gammar):
                    delta_j_t=1,0
                else:
                    delta_j_t=0,1
                return delta_j_t
            delta_result={(j,t):judge(j,t) for j in range(1,self._routeNo+1) for t in range(1,self._period+1)}
            return delta_result

        def headway(x,delta_j_t_s):
            #delta_j_t_s=delta(x)
            def headway_j_t_s(j,t,s):
                temp1=[2*self._gammar*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*delta_j_t_s[j,t][s-1]/self._speed[j-1][t-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1)]
                temp1=np.sum(temp1)
                temp2=[self._v_w*self._demand[j-1][t-1]*delta_j_t_s[j,t][s-1]/self._peak_point_demand[j-1][t-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1)]
                temp2=np.sum(temp2)
                headway1_j_t_s=np.sqrt(temp1/temp2)/self._peak_point_demand[j-1][t-1]
                headway2_j_t_s=2*self._distance[j-1]*(self._gammar+self._beta*x[s-1])/(self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._v_w)
                Headway_j_t_s=np.min([headway1_j_t_s,headway2_j_t_s])
                return Headway_j_t_s
            headway_result={(j,t):tuple(headway_j_t_s(j,t,s) for s in range(1,self._size_type+1)) for j in range(1,self._routeNo+1) for t in range(1,self._period+1)}
            return headway_result

        def fleet_size(x,delta_jts,headway_jts):
            def N_ts(t,s):
                temp1=[2*self._distance[j-1]*delta_jts[j,t][s-1]/self._speed[j-1][t-1]/headway_jts[j,t][s-1] if headway_jts[j,t][s-1]!=0 else 0 for j in range(1,self._routeNo+1)]
                temp1=np.sum(temp1)
                return temp1
            N_t=[tuple(N_ts(t,s) for s in range(1,self._size_type+1)) for t in range(self._period+1)]
            N_t_temp=[[N_t[j][i] for j in range(len(N_t))] for i in range(self._size_type+1)]
            N_s=np.max(N_t_temp,axis=1)
            N_s=tuple(N_s)
            return N_s

        def ctp(x,delta,fleet_size):
            c_o_temp=[(2*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*(self._gammar+self._beta*x[s-1])/self._speed[j-1][t-1]/x[s-1])*delta[j,t][s-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1) for s in range(1,self._size_type+1)]
            c_uw_temp=[self._v_w*self._demand[j-1][t-1]*x[s-1]/self._peak_point_demand[j-1][t-1]*delta[j,t][s-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1) for s in range(1,self._size_type+1)]
            c_uv_temp=[(2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1])*delta[j,t][s-1] for j in range(1,self._routeNo+1) for t in range(1,self._period+1) for s in range(1,self._size_type+1)]
            c_o=np.sum(c_o_temp)
            c_uw=np.sum(c_uw_temp)
            c_uv=np.sum(c_uv_temp)
            c_u=c_uw+c_uv
            c_total=c_o+c_u
            c_p_temp=[((self._c+self._e*x[s-1])*self._recovery/365)*fleet_size[s-1] for s in range(1,self._size_type+1)]
            c_p=np.sum(c_p_temp)
            c_tp=c_total+c_p
            return c_tp,c_o,c_uw,c_uv,c_u,c_total,c_p

        def num_grad(x,delta,fleet_size):
            x=list(x)
            def num_grad_item(h,i):
                x1=copy.copy(x)
                x2=copy.copy(x)
                x1[i]=x1[i]-h
                x2[i]=x2[i]+h
                y1,y2=ctp(x1,delta,fleet_size)[0],ctp(x2,delta,fleet_size)[0]
                df=(y2-y1)/(2*h)
                return df
            df=[num_grad_item(10**-5,i) for i in range(len(x))]
            df=np.array(df)
            return df

        def num_hess(x,delta,fleet_size):
            x=list(x)
            def hess_row(h,i):
                x1=copy.copy(x)
                x1[i]=x1[i]-h
                df1=num_grad(x1,delta,fleet_size)
                x2=copy.copy(x)
                x2[i]=x2[i]+h
                df2=num_grad(x2,delta,fleet_size)
                d2f=(df2-df1)/(2*h)
                return d2f
            hess=[hess_row(10**-5,i) for i in range(len(x))]
            hess=np.array(hess)
            return hess

        def linesearch(x,dk,delta,fleet_size):
            x=np.array(x)
            ak=1
            for i in range(20):
                newf,oldf=ctp(x+ak*dk,delta,fleet_size)[0],ctp(x,delta,fleet_size)[0]
                if newf<oldf:
                    return ak
                else:
                    ak=ak/4
            return ak
        

