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
sb.set()

class OneSize(object):
    def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand):
        self._beta=0.25
        self._gammmar=25
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
            m=gp.Model('OneSizeSystemwide')

            index_stop_period=gp.tuplelist([(x,y) for x in range(1,self._routeNo+1) for y in range(1,self._period+1)])

            size_volume=m.addVar(name='size_volume')
            bus_operating=m.addVar(name='bus_operating')
            h_jt=m.addVars(index_stop_period,name='headway_jt')
            h_jt_1=m.addVars(index_stop_period,name='headway1_jt')
            h_jt_2=m.addVars(index_stop_period,name='headway2_jt')
            c_o_j_t=m.addVars(index_stop_period,name='cost_operator_jt')
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
            n_jt=m.addVars(index_stop_period,name='fleet_size_jt')
            n_t=m.addVars(range(1,self._period+1),name='fleet_size_t')
            n=m.addVar(name='fleet_size')
            m.update()

            m.setObjective(c_total+c_p, sense=gp.GRB.MINIMIZE)

            m.addConstr(bus_operating==self._gammar+self._beta*size_volume,name='bus_operating_cost')
            m.addConstrs((c_o_j_t[j,t]==2*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*bus_operating/self._speed[j-1][t-1]/size_volume for j,t in index_stop_period),name="cost_operator_jt_c")
            m.addConstrs((c_uw_j_t[j,t]==self.v_w*self._demand[j-1][t-1]*size_volume/self._peak_point_demand[j-1][t-1] for j,t in index_stop_period),name='cost_user_waiting_jt_c')
            m.addConstrs((c_uv_j_t[j,t]==self.v_v*2*self._average_distance[j-1]*self._demand[j-1][t-1]/self._speed[j-1][t-1] for j,t in index_stop_period),name='cost_user_invehicle_jt_c')
            m.addConstrs((c_u_j_t[j,t]==c_uw_j_t[j,t]+c_uv_j_t[j,t] for j,t in index_stop_period),name='cost_user_jt_c')
            m.addConstrs((c_j_t[j,t]==c_u_j_t[j,t]+c_o_j_t[j,t] for j,t in index_stop_period),name="cost_jt_c")
            m.addConstr(c_o==gp.quicksum(c_o_j_t[j,t] for j,t in index_stop_period),name='cost_operator_c')
            m.addConstr(c_uw==gp.quicksum(c_uw_j_t[j,t] for j,t in index_stop_period),name='cost_user_waiting_c')
            m.addConstr(c_uv==gp.quicksum(c_uv_j_t[j,t] for j,t in index_stop_period),name='cost_user_invehicle_c')
            m.addConstr(c_u==gp.quicksum(c_u_j_t[j,t] for j,t in index_stop_period),name='cost_user_c')
            m.addConstr(c_total==gp.quicksum(c_j_t[j,t] for j,t in index_stop_period),name='cost_total_c')
            m.addConstrs((h_jt_1[j,t]==size_volume/self._peak_point_demand[j-1][t-1] for j,t in index_stop_period),name='headway1_jt_c')
            m.addConstrs((h_jt_2[j,t]==2*self._distance[j-1]*bus_operating/self._speed[j-1][t-1]/self._demand[j-1][t-1]/self.v_w for j,t in index_stop_period),name='headway2_jt_c')
            m.addConstrs((h_jt[j,t]==max_(h_jt_1[j,t],h_jt_2[j,t]) for j,t in index_stop_period),name='headway_jt_c')
            m.addConstrs((n_jt[j,t]==2*self._distance[j-1]/self._speed[j-1][t-1]/h_jt[j,t] for j,t in index_stop_period),name='fleet_size_jt_c')
            m.addConstrs((n_t[t]==gp.quicksum(n_jt.select('*',t)) for t in range(1,self._period+1)),name='fleet_size_t_c')
            m.addGenConstrMax(n,[n_t[t] for t in self._period+1],name='fleet_size_c')
            m.addConstr(c_p==(self.c+self.e*size_volume)*self.recovery_rate/365*n,name='cost_capital_c')

        except gp.GurrobiError as e:
            print('Error code'+str(e.errno)+': '+str(e))
        except AttributeError:
            print('Encountered an attribute error')
