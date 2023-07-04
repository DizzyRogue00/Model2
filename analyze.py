import FOT as fot
import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os
import logging.config
import yaml
import math
from matplotlib import cm
import matplotlib.ticker as ticker

sb.set()


with open('logging.yaml','r',encoding='utf-8') as f:
    #config=yaml.load(f,Loader=yaml.FullLoader)
    config = yaml.safe_load(f)
    #print(config)
    logging.config.dictConfig(config)
logger=logging.getLogger('mylogger')

def load_pickle(path):
    with open(path,'rb') as f:
        while 1:
            try:
                y=pickle.load(f)
                yield y
            except EOFError:
                logger.error('Error', exc_info=True)
                break

with open('data_one.pickle', 'rb') as f:
    result_one=pickle.load(f)
with open('data_two.pickle', 'rb') as f:
    result_two=pickle.load(f)

result_baseline=load_pickle('data_alpha_1.2_demand_1.pickle')
result_baseline=list(result_baseline)
print('OK')


#plt.figure(num=1,facecolor='white',edgecolor='black')
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
data_covergence={}
data_covergence['UBD']=[result_baseline[-1][i][0] for i in result_baseline[-1].keys()]
data_covergence['LBD']=[result_baseline[-1][i][1] for i in result_baseline[-1].keys()]
max_data=max(data_covergence['UBD'])
data_covergence['UBD']=[math.log(i)/math.log(max_data) for i in data_covergence['UBD']]
data_covergence['LBD']=[math.log(i)/math.log(max_data) for i in data_covergence['LBD']]
#data_covergence['UBD'].insert(0,float('inf'))
#data_covergence['LBD'].insert(0,float('inf'))
data_convergence=pd.DataFrame(data_covergence,index=range(1,len(result_baseline[-1])+1))
markers_ZC=[".","^"]
linestyle=['-','--']
#color = ['#7B113A', "#150E56", "#1597BB", "#8FD6E1", "#E02401", "#F78812", "#Ab6D23", "#51050F"]
color = ['#7B113A', "#150E56"]
style=list(map(lambda x,y:x+y,markers_ZC,linestyle))
data_convergence.plot(style=style,color=color,legend=True,linewidth=1.5)

ax = plt.gca()
ax.tick_params(top=False,bottom=True,left=True,right=False,direction='inout')
ax.tick_params(which='major',width=1.5)
ax.set_facecolor("w")
ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
plt.xticks(range(1, len(result_baseline[-1]) + 1), fontweight='bold')
plt.yticks(None,None,fontweight='bold')
#plt.yticks(np.array([0,0.2,0.4,0.6,0.8,1.0])*1e5, None, fontweight='bold')
#plt.yticks(np.arange(0.76,1.2,step=0.1), None, fontweight='bold')
#plt.ylim((0.75,1))
plt.xlabel("Iteration No.", fontdict=dict(fontweight='bold'))
plt.ylabel('Optimal Value (scaled)', fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.savefig('Results/Algorithm convergence.pdf', dpi=1000,format='pdf')
plt.savefig('Results/ALgorithm convergence.png', dpi=1000,format='png')
#plt.show(block=True)

distance = [16, 20, 16, 24, 20]
average_distance = [5, 7, 5, 8, 6]
speed=[
        [48,40,40,48,48,48,48,48,48,48,48,40,40,48,48,48,48,48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48]
    ]
demand=[
        [100,200,200,100,100,100,100,100,100,100,100,200,200,100,100,100,100,100],
        [150, 300, 300, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 150, 150, 150, 150, 150],
        [60, 150, 150, 60, 60, 60, 60, 60, 60, 60, 60, 150, 150, 60, 60, 60, 60,60],
        [200, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 200, 200, 200, 200, 200],
        [125, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 250, 250, 125, 125, 125, 125, 125]
    ]
peak_point_demand=[
        [100,200,200,100,100,100,100,100,100,100,100,200,200,100,100,100,100,100],
        [150, 300, 300, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 150, 150, 150, 150, 150],
        [60, 150, 150, 60, 60, 60, 60, 60, 60, 60, 60, 150, 150, 60, 60, 60, 60,60],
        [200, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 200, 200, 200, 200, 200],
        [125, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 250, 250, 125, 125, 125, 125, 125]
    ]
gamma=25 #$/veh.hr
beta=0.25 #$/veh.seat.hr
v_w=10    #time value of waiting time $/pax.hr
v_v=6     #time value of in-vehicle time $/pax.hr
t_u=1.0/180   #hr/parcel
c=16000   #$/veh
e=2400    #$/veh
recovery=0.1359
eta=0.125
v_p=1500

sum_o=0
sum_uw=0
sum_uv=0
sum_p=0
sum_dr=0
data_x=result_baseline[-3]
data_y=result_baseline[-2]
for j in range(1,6):
    for t in range(1,19):
        sum_o=sum_o+2*distance[j-1]*(gamma+beta*data_x['S'][1])*peak_point_demand[j-1][t-1]/data_x['v_hat'][j,t]/data_x['S'][1]*data_y['X'][j,t]*data_y['delta'][j,t]
        sum_o=sum_o+2*distance[j-1]*(gamma+beta*data_x['S'][1])*peak_point_demand[j-1][t-1]/speed[j-1][t-1]/data_x['v_hat'][j,t]*(1-data_y['X'][j,t])*data_y['delta'][j,t]
        sum_o=sum_o+2*distance[j-1]*(gamma+beta*data_x['S'][2])*peak_point_demand[j-1][t-1]/speed[j-1][t-1]/data_x['S'][2]*(1-data_y['delta'][j,t])

        sum_uw=sum_uw+v_w*demand[j-1][t-1]*data_x['S'][1]/peak_point_demand[j-1][t-1]*data_y['delta'][j,t]
        sum_uw=sum_uw+v_w*demand[j-1][t-1]*data_x['S'][2]/peak_point_demand[j-1][t-1]*(1-data_y['delta'][j,t])

        sum_uv=sum_uv+v_v*peak_point_demand[j-1][t-1]*2*average_distance[j-1]/data_x['v_hat'][j,t]*data_y['X'][j,t]*data_y['delta'][j,t]
        sum_uv=sum_uv+v_v*peak_point_demand[j-1][t-1]*2*average_distance[j-1]/speed[j-1][t-1]*(1-data_y['X'][j,t]*data_y['delta'][j,t])

for s in range(1,3):
    sum_p=sum_p+(c+e*data_x['S'][s])*recovery/365*data_y['N_bar'][s]

freight_demand=[175,113,193,233,132]
for j in range(1,6):
    for t in range(1,19):
        sum_dr=sum_dr+data_y['q'][j,t]
sum_dr=sum(freight_demand)-sum_dr
sum_dr=max(0,sum_dr)
sum_dr=sum_dr*v_p

scatter_x=[list(np.arange(6.5,24,step=1))]*3*5
scatter_y=[[i]*18 for i in np.arange(15,0,step=-1)]
scatter_color=[]
for j in range(1,6):
    j_x=[]
    j_delta=[]
    j_q=[]
    for t in range(1,19):
        if data_y['X'][j,t]==1:
            j_x.append('firebrick')
        else:
            j_x.append('w')
        if data_y['delta'][j,t]==1:
            j_delta.append('#7BA23F')
        else:
            j_delta.append('w')
        j_q.append(data_y['q'][j,t])
    scatter_color.append(j_q)
    scatter_color.append(j_x)
    scatter_color.append(j_delta)
fig,ax=plt.subplots(figsize=(16,10),dpi=300)
ax.hlines(y=range(1,16),xmin=6,xmax=24,color='gray',alpha=0.5,linewidth=.5,linestyles='dashdot')
for j in range(1,6):
    ax.scatter(x=scatter_x[3*j-1],y=scatter_y[3*j-1],c=scatter_color[3*j-1],linewidths=0.5,edgecolors='gray',s=50)
    ax.scatter(x=scatter_x[3*j-2],y=scatter_y[3*j-2],c=scatter_color[3*j-2],linewidths=0.5,edgecolors='gray',s=50)
    sc=ax.scatter(x=scatter_x[3 * j -3], y=scatter_y[3 * j-3], c=scatter_color[3 * j-3],cmap='Blues',linewidths=0.5,edgecolors='gray',s=50)
cb=plt.colorbar(sc)
cb.ax.tick_params(labelsize=5,direction='inout')
# tick_locator=ticker.MaxNLocator(nbins=7)
# cb.locator=tick_locator
# cb.set_ticks([0,5,10,15,20,25,30])
# cb.update_ticks()
# norm=plt.Normalize(0,30)
# sm=plt.cm.ScalarMappable(cmap='Blues')
# sm.set_array([])
# #ax.get_legend().remove()
# ax.figure.colorbar(sm)
plt.plot([],[],marker="o", ms=5, ls="", mec=None, color='firebrick',label=r"$X$")
plt.plot([],[],marker="o", ms=5, ls="", mec=None, color='#7BA23F',label=r"$\delta$")
plt.legend(loc='upper right',fontsize=5,bbox_to_anchor=(1.065,1),markerscale=1.,framealpha=0.5,frameon=False,facecolor='white')
ax.set_xlabel('Time',fontsize=5,fontweight='bold')
ax.set_ylabel('Bus Route',fontsize=5,fontweight='bold')
ax.set_xticks(np.arange(6,24,step=1))
ax.set_yticks(range(1,16))
ax.set_xticklabels(np.arange(6,24,step=1),fontdict=dict(fontweight='bold'),alpha=0.7,fontsize=5)
ax.set_yticklabels([5,5,5,4,4,4,3,3,3,2,2,2,1,1,1],fontdict=dict(fontweight='bold'),alpha=0.7,fontsize=5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis='both', alpha=.4, linewidth=.1)
#plt.show(block=True)
plt.tight_layout()
plt.savefig('Results/Bus dispatch schedule.pdf', dpi=1000,format='pdf')
plt.savefig('Results/Bus dispatch schedule.png', dpi=1000,format='png')
#plt.show(block=True)

demand_rate=list(np.arange(5,21)/10)

def draw_schedule_HSN(routeNo, distance, average_distance, speed, demand, peak_point_demand, rate_demand,rate_speed=1.2):
    case = fot.FOT(routeNo, distance, average_distance, speed, demand, peak_point_demand, rate_speed, rate_demand)
    case_d=case._d_j
    period = len(demand[0])
    y_initial = {}
    y_initial['y_0'] = -float("inf")
    y_initial['N_hat'] = {(j, t): 10 for j in range(1, routeNo + 1) for t in range(1, period + 1)}
    y_initial['N_bar'] = {1: 100, 2: 100}
    y_initial['q'] = {(j, t): 0 for j in range(1, routeNo + 1) for t in range(1, period + 1)}
    y_initial['X'] = {(j, t): 0 for j in range(1, routeNo + 1) for t in range(1, period + 1)}
    y_initial['delta'] = {(j, t): 0 for j in range(1, routeNo + 1) for t in range(1, period + 1)}
    y_initial['xi'] = {(j, t): 0 for j in range(1, routeNo + 1) for t in range(1, period + 1)}
    y_initial['zeta'] = {(j, t): 0 for j in range(1, routeNo + 1) for t in range(1, period + 1)}

    result = case.solveBenders(100, y_initial, 1000)

    sum_o = 0
    sum_uw = 0
    sum_uv = 0
    sum_p = 0
    sum_dr = 0
    data_x = result[0]
    data_y = result[1]
    for j in range(1, routeNo+1):
        for t in range(1, period+1):
            sum_o = sum_o + 2 * distance[j - 1] * (case.gammar + case.beta * data_x['S'][1]) * peak_point_demand[j - 1][t - 1] / \
                    data_x['v_hat'][j, t] / data_x['S'][1] * data_y['X'][j, t] * data_y['delta'][j, t]
            sum_o = sum_o + 2 * distance[j - 1] * (case.gamma + case.beta * data_x['S'][1]) * peak_point_demand[j - 1][t - 1] / \
                    speed[j - 1][t - 1] / data_x['v_hat'][j, t] * (1 - data_y['X'][j, t]) * data_y['delta'][j, t]
            sum_o = sum_o + 2 * distance[j - 1] * (case.gamma + case.beta * data_x['S'][2]) * peak_point_demand[j - 1][t - 1] / \
                    speed[j - 1][t - 1] / data_x['S'][2] * (1 - data_y['delta'][j, t])

            sum_uw = sum_uw + case.v_w * demand[j - 1][t - 1] * data_x['S'][1] / peak_point_demand[j - 1][t - 1] * \
                     data_y['delta'][j, t]
            sum_uw = sum_uw + case.v_w * demand[j - 1][t - 1] * data_x['S'][2] / peak_point_demand[j - 1][t - 1] * (
                        1 - data_y['delta'][j, t])

            sum_uv = sum_uv + case.v_v * peak_point_demand[j - 1][t - 1] * 2 * average_distance[j - 1] / data_x['v_hat'][
                j, t] * data_y['X'][j, t] * data_y['delta'][j, t]
            sum_uv = sum_uv + case.v_v * peak_point_demand[j - 1][t - 1] * 2 * average_distance[j - 1] / speed[j - 1][
                t - 1] * (1 - data_y['X'][j, t] * data_y['delta'][j, t])

    for s in range(1, 3):
        sum_p = sum_p + (case.c + case.e * data_x['S'][s]) * case.recovery / 365 * data_y['N_bar'][s]

    for j in range(1, routeNo+1):
        for t in range(1, period+1):
            sum_dr = sum_dr + data_y['q'][j, t]
    sum_dr = sum(case_d) - sum_dr
    sum_dr = max(0, sum_dr)
    dr_ratio=sum_dr/sum(case_d)*100
    sum_dr = sum_dr * case.v_p

    N_bar=result[1]['N_bar']
    S=result[0]['S']
    H=result[0]['headway']
    H_peak=[]
    H_off_peak=[]
    for j in range(1,routeNo+1):
        H_j=[H[item1,item2] for item1,item2 in H.keys() if item1==j]
        index=[1,2,11,12]
        H_j_1=[item2 for (item1,item2) in enumerate(H_j) if item1 in index]
        H_j_2=[item2 for (item1,item2) in enumerate(H_j) if item1 not in index]
        H_j_1_max=max(H_j_1)
        H_j_2_max=max(H_j_2)
        H_peak.append(H_j_1_max)
        H_off_peak.append(H_j_2_max)
    H_peak_mean=np.mean(H_peak)
    H_off_peak_mean=np.mean(H_off_peak)

    if rate_demand/0.5 in [1.0,2.0,3.0,4.0]:
        scatter_x = [list(np.arange(6.5, 24, step=1))] * 3 *routeNo
        scatter_y = [[i] * period for i in np.arange(3*routeNo, 0, step=-1)]
        scatter_color = []
        for j in range(1, routeNo+1):
            j_x = []
            j_delta = []
            j_q = []
            for t in range(1, period+1):
                if data_y['X'][j, t] == 1:
                    j_x.append('firebrick')
                else:
                    j_x.append('w')
                if data_y['delta'][j, t] == 1:
                    j_delta.append('#7BA23F')
                else:
                    j_delta.append('w')
                j_q.append(data_y['q'][j, t])
            scatter_color.append(j_q)
            scatter_color.append(j_x)
            scatter_color.append(j_delta)
        fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
        ax.hlines(y=range(1, 16), xmin=6, xmax=24, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')
        for j in range(1, routeNo+1):
            ax.scatter(x=scatter_x[3 * j - 1], y=scatter_y[3 * j - 1], c=scatter_color[3 * j - 1], linewidths=0.5,
                       edgecolors='gray', s=50)
            ax.scatter(x=scatter_x[3 * j - 2], y=scatter_y[3 * j - 2], c=scatter_color[3 * j - 2], linewidths=0.5,
                       edgecolors='gray', s=50)
            sc = ax.scatter(x=scatter_x[3 * j - 3], y=scatter_y[3 * j - 3], c=scatter_color[3 * j - 3], cmap='Blues',
                            linewidths=0.5, edgecolors='gray', s=50)
        cb = plt.colorbar(sc)
        cb.ax.tick_params(labelsize=5, direction='inout')
        plt.plot([], [], marker="o", ms=5, ls="", mec=None, color='firebrick', label=r"$X$")
        plt.plot([], [], marker="o", ms=5, ls="", mec=None, color='#7BA23F', label=r"$\delta$")
        plt.legend(loc='upper right', fontsize=5, bbox_to_anchor=(1.065, 1), markerscale=1., framealpha=0.5,
                   frameon=False, facecolor='white')
        ax.set_xlabel('Time', fontsize=5, fontweight='bold')
        ax.set_ylabel('Bus Route', fontsize=5, fontweight='bold')
        ax.set_xticks(np.arange(6, 24, step=1))
        ax.set_yticks(range(1, 16))
        ax.set_xticklabels(np.arange(6, 24, step=1), fontdict=dict(fontweight='bold'), alpha=0.7, fontsize=5)
        ax.set_yticklabels([5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1], fontdict=dict(fontweight='bold'), alpha=0.7,
                           fontsize=5)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.grid(axis='both', alpha=.4, linewidth=.1)
        # plt.show(block=True)
        plt.tight_layout()
        name_pdf='Results/Bus dispatch schedule_speedRate_1.2_demandRate_'+str(rate_demand)+'pdf'
        name_png = 'Results/Bus dispatch schedule_speedRate_1.2_demandRate_' + str(rate_demand) + 'png'
        plt.savefig(name_pdf, dpi=1000, format='pdf')
        plt.savefig(name_png, dpi=1000, format='png')

    return case_d,dr_ratio,sum_o,sum_uw,sum_uv,sum_p,sum_dr,N_bar,S,H_peak,H_off_peak,(H_peak_mean,H_off_peak_mean)

analy={}
analy['dr_ratio']=[]
analy['N_bar1']=[]
analy['N_bar2']=[]
analy['S1']=[]
analy['S2']=[]
analy['H_peak']=[]
analy['H_off_peak']=[]

for item in demand_rate:
    result=draw_schedule_HSN(5, distance, average_distance, speed, demand, peak_point_demand, item,rate_speed=1.2)
    analy['dr_ratio'].append(result[1])
    analy['N_bar1'].append(result[7][1])
    analy['N_bar2'].append(result[7][2])
    analy['S1'].append(result[8][1])
    analy['S2'].append(result[8][2])
    analy['H_peak'].append(result[11][0])
    analy['H_off_peak'].append(result[11][1])
my_df=pd.DataFrame(analy,index=demand_rate)
my_df.to_csv('Sensitive analysis on demand')