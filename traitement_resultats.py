from re import T
from matplotlib.pyplot import clabel
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

def find_solution(df, x, y, z): # x coef chauffage, y coef confort, z coef cout
    """To find intermediate solution"""
    chauff=df["besoins de chauffage kWh/m2"]
    inconf=df["heures d'inconfort (T>Tconf+2°C)"]
    cout=df["Cout global actualisé en euros/m2"]
    chauff_min=chauff.min()
    chauff_max=chauff.max()
    inconf_min=inconf.min()
    inconf_max=inconf.max()
    cout_min=cout.min()
    cout_max=cout.max()
    objectif=(x*(chauff-chauff_min)/(chauff_max-chauff_min)+y*(inconf-inconf_min)/(inconf_max-inconf_min)+z*(cout-cout_min)/(cout_max-cout_min))
    #print(objectif)
    df["objectif"]=objectif
    objectif_min=df["objectif"].min()
    index_objectif_min=df[df["objectif"]==objectif_min].index.values
    solution=df.iloc[index_objectif_min]
    solution=solution.values.tolist()
    print("la solution intermediaire", x, y, z , "est\n", solution)
    return solution
def find_solution_KS(df): 
    """Trouver la solution la plus proche de l'origine (tout est normalisé) car sinon la solution est différente"""
    #To find the Kalai–Smorodinsky bargaining solution
    chauff=df["besoins chauffage"]
    inconf=df["inconfort"]
    cout=df["cout actualisé"]
    chauff_min=chauff.min()
    chauff_max=chauff.max()
    inconf_min=inconf.min()
    inconf_max=inconf.max()
    cout_min=cout.min()
    cout_max=cout.max()
    chauff_norm=(chauff-chauff_min)/(chauff_max-chauff_min)
    inconf_norm=(inconf-inconf_min)/(inconf_max-inconf_min)
    cout_norm=(cout-cout_min)/(cout_max-cout_min)
    objectif=chauff_norm**2+inconf_norm**2+cout_norm**2
    utopia=[chauff.min(),inconf.min(),cout.min()]
    nadir=[chauff.max(),inconf.max(),cout.max()]
    df["KS"]=objectif
    KS_min=df["KS"].min()
    index_KS_min=df[df["KS"]==KS_min].index.values
    solution=df.iloc[index_KS_min]
    solution=solution.values.tolist()
    return solution
def find_best_solution(df): 
    """Trouver la solution la plus proche de l'origine (sans normalisation)"""
    chauff=df["besoins chauffage"]
    inconf=df["inconfort"]
    cout=df["cout actualisé"]
    objectif=chauff**2+inconf**2+cout**2
    df["KS"]=objectif
    KS_min=df["KS"].min()
    index_KS_min=df[df["KS"]==KS_min].index.values
    solution=df.iloc[index_KS_min]
    solution=solution.values.tolist()
    return solution
def find_solution_utopia(df): 
    """Trouver la solution la plus proche du point utopia (sans normalisation)"""
    chauff=df["besoins de chauffage kWh/m2"]
    inconf=df["heures d'inconfort (T>Tconf+2°C)"]
    cout=df["Cout global actualisé en euros/m2"]
    chauff_min=chauff.min()
    inconf_min=inconf.min()
    cout_min=cout.min()
    objectif=(chauff-chauff_min)**2+(inconf-inconf_min)**2+(cout-cout_min)**2
    utopia=[chauff.min(),inconf.min(),cout.min()]
    nadir=[chauff.max(),inconf.max(),cout.max()]
    df["KS"]=objectif
    KS_min=df["KS"].min()
    index_KS_min=df[df["KS"]==KS_min].index.values
    solution=df.iloc[index_KS_min]
    solution=solution.values.tolist()
    return solution
def sort_solutions(df, x, y, z): # x coef chauffage, y coef confort, z coef cout
    """To find intermediate solution"""
    chauff=df["besoins de chauffage kWh/m2"]
    inconf=df["heures d'inconfort (T>Tconf+2°C)"]
    cout=df["Cout global actualisé en euros/m2"]
    chauff_min=chauff.min()
    chauff_max=chauff.max()
    inconf_min=inconf.min()
    inconf_max=inconf.max()
    cout_min=cout.min()
    cout_max=cout.max()
    objectif=(x*(chauff-chauff_min)/(chauff_max-chauff_min)+y*(inconf-inconf_min)/(inconf_max-inconf_min)+z*(cout-cout_min)/(cout_max-cout_min))
    #print(objectif)
    df["objectif"]=objectif
    df["rank"]=df["objectif"].rank()
    df.sort_values("objectif", inplace = True)
    df.to_excel("sort.xlsx")
    return df
def plots_grouped(df):   
    x = df["besoins de chauffage kWh/m2"]#chauffage
    y = df["Cout global actualisé en euros/m2"] #cout
    z = df["heures d'inconfort (T>Tconf+2°C)"] #inconfort
    ep_mur=df["ep_murs_ext"]
    ep_ph=df["ep_plancher_haut"]
    ep_pb=df["ep_plancher_bas"]
    vitrage=df["type_fenetre"]
    fig,axes=plt.subplots(nrows=2,ncols=2, sharey=True)
    fig.set_size_inches(15,10)
    axes[0,0].set_ylabel('Cout actualisé en euros/m2')
    axes[1,0].set_ylabel('Cout actualisé en euros/m2')
    plot1=axes[0,0].scatter(x, y, c=vitrage,s=z-100, alpha=0.5)
    #axes[1,0].set_ylabel("Heures d'inconfort")
    axes[1,0].set_xlabel("Besoins de chauffage kWh/m2")
    plot2=axes[1,0].scatter(x, y, c=ep_mur,s=z-100, alpha=0.5)
    #axes[1,1].set_xlabel("Cout actualisé en euros/m2")
    plot3=axes[1,1].scatter(x, y, c=ep_pb,s=z-100, alpha=0.5)
    plot4=axes[0,1].scatter(x, y, c=ep_ph,s=z-100, alpha=0.5)
    legend1 = axes[0,0].legend(*plot1.legend_elements(),loc="lower left", title="type_vitrage")
    axes[0,0].add_artist(legend1)
    handles, labels = plot1.legend_elements(prop="sizes", alpha=0.6)
    legend = axes[0,0].legend(handles, labels, loc="upper right", title="heures d'inconfort-100")
    legend2 = axes[1,0].legend(*plot2.legend_elements(),loc="upper right", title="ep_murs_ext")
    legend3 = axes[1,1].legend(*plot3.legend_elements(),loc="upper right", title="ep_plancher_haut")
    legend4 = axes[0,1].legend(*plot4.legend_elements(),loc="upper right", title="ep_plancher_bas")
    plt.show()
def plots_3d_grouped(df):
    x = df["besoins de chauffage kWh/m2"]#chauffage
    y = df["Cout global actualisé en euros/m2"] #cout
    z = df["heures d'inconfort (T>Tconf+2°C)"] #inconfort
    ep_mur=df["ep_murs_ext"]
    ep_ph=df["ep_plancher_haut"]
    ep_pb=df["ep_plancher_bas"]
    vitrage=df["type_fenetre"]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot1=ax.scatter(x, y, z, c=vitrage, alpha=0.5)
    ax.set_xlabel("besoins de chauffage kWh/m2")
    ax.set_ylabel("Cout actualisé en euros/m2")
    ax.set_zlabel("heures d'inconfort")
    #legend1 = ax.legend(*plot1.legend_elements(),loc="lower left", title="type_vitrage")
    #fig.legend()
    fig.show()
def plots_2d_grouped(df):
    x = df["besoins de chauffage kWh/m2"]#chauffage
    y = df["Cout global actualisé en euros/m2"] #cout
    z = df["heures d'inconfort (T>Tconf+2°C)"] #inconfort
    ep_mur=df["ep_murs_ext"]
    ep_ph=df["ep_plancher_haut"]
    ep_pb=df["ep_plancher_bas"]
    vitrage=df["type_fenetre"]
    fig = plt.figure()
    #fig.set_size_inches(15,10)    
    axe1 = plt.subplot2grid((2,2),(0,0))
    axe1.set_ylabel('Cout actualisé en euros/m2')
    plot1=axe1.scatter(x, y, c=vitrage, alpha=0.5)
    legend1 = axe1.legend(*plot1.legend_elements(),loc="upper right", title="type_vitrage")
    axe2 = plt.subplot2grid((2,2),(1,0))
    axe2.set_ylabel("Heures d'inconfort (T>Tconf+2°C)")
    axe2.set_xlabel("Besoins de chauffage kWh/m2")
    plot2=axe2.scatter(x, z, c=vitrage, alpha=0.5)
    axe3 = plt.subplot2grid((2,2),(1,1))
    axe3.set_xlabel("Cout actualisé en euros/m2")
    plot3 = axe3.scatter(y, z, c=vitrage, alpha=0.5)
    plt.show()
def plots(df, df2=None, base_solution=None,KS=None, plot2D = True, plot3D = False, interactive = False, label=""):   
    x = df["besoins de chauffage kWh/m2"]#chauffage
    y = df["Cout global actualisé en euros/m2"] #cout
    z = df["heures d'inconfort (T>Tconf+2°C)"] #inconfort
    if df2 is None:
        df2=pd.DataFrame()  
    if not df2.empty:
            x2 = df2["besoins de chauffage kWh/m2"]#chauffage
            y2 = df2["Cout global actualisé en euros/m2"] #cout
            z2 = df2["heures d'inconfort (T>Tconf+2°C)"] #inconfort
    economic_solution=find_solution(df, 0, 0, 1)
    comfortable_solution=find_solution(df, 0, 1, 0)
    efficient_solution=find_solution(df, 1, 0, 0)
    compromis_solution=find_solution(df, 1/3, 1/3, 1/3) 
    utopia_solution=find_solution_utopia(df)
    print ("utopia:",utopia_solution)
    #KS_solution=find_solution_KS(df_nomass)  
    if plot3D == True:
        """to plot the tree functions"""
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, alpha=0.5)
        if base_solution != None: #.all()
            ax.scatter(base_solution[0], base_solution[2], base_solution[1], marker="s", color="purple",s=50, label="Cas de référence")
        ax.scatter(economic_solution[0][0], economic_solution[0][2], economic_solution[0][1], marker="s", s=50,color="r", label="Solution 1")
        ax.scatter(efficient_solution[0][0], efficient_solution[0][2], efficient_solution[0][1],marker="s",s=50, color="k", label="Solution 2")
        ax.scatter(comfortable_solution[0][0], comfortable_solution[0][2], comfortable_solution[0][1], marker="s",s=50, color="y", label="Solution 3")
        ax.scatter(compromis_solution[0][0], compromis_solution[0][2],compromis_solution[0][1], marker="s",s=50, color="g", label="Solution 4")
        if KS != None: #.all()
            ax.scatter(KS[0], KS[2], KS[1], marker="s", color="cyan",s=50, label="Solution 5")
        ax.scatter(utopia_solution[0][0], utopia_solution[0][2],utopia_solution[0][1], marker="s",s=50, color="grey", label="Solution 6")
        if not df2.empty:
            ax.scatter(x2, y2,z2, c='b', alpha=0.5)
        ax.set_xlabel("besoins de chauffage kWh/m2")
        ax.set_ylabel("Cout actualisé en euros/m2")
        ax.set_zlabel("heures d'inconfort (T>Tconf+2°C)")
        fig.legend()
        ax.figure.savefig('./graphes/'+label+'_3D.png')
        fig.show() 
    if plot2D == True : 
        """to plot 2 by 2"""

        '''
        fig = plt.figure()
        plt.scatter(x,y,c=z)
        plt.xlabel("Besoins de chauffage kWh")
        plt.ylabel('Cout actualisé en euros')
        plt.title('Front de Pareto')
        plt.colorbar(label="Heures d'inconfort (T>Tconf+2°C)")
        plt.savefig('./graphes/Front de Pareto_objectifs.png')
        '''
        
        fig = plt.figure()
        fig.set_size_inches(15,10)
        
        axe1 = plt.subplot2grid((2,2),(0,0))
        axe1.set_ylabel('Cout actualisé en euros/m2')
        plot1=axe1.scatter(x, y, c=z, alpha=0.5)
        axe1.scatter(economic_solution[0][0], economic_solution[0][2], marker="s", color="r",s=50, label="solution 1")
        axe1.scatter(efficient_solution[0][0], efficient_solution[0][2], marker="s", color="k",s=50, label="solution 2")
        axe1.scatter(comfortable_solution[0][0], comfortable_solution[0][2], marker="s", color="y",s=50, label="solution 3")
        axe1.scatter(compromis_solution[0][0], compromis_solution[0][2], marker="s", color="g",s=50, label="solution 4")
        axe1.scatter(utopia_solution[0][0], utopia_solution[0][2], marker="s",s=50, color="grey", label="Solution 6")
        #axe1.scatter(KS_solution[0][0], KS_solution[0][2], marker="s", color="b",s=50, label="l'équilibre de Kalai-Smorodinsky")
        plt.colorbar(plot1,ax=axe1,label="Heures d'inconfort (T>Tconf+2°C)")


        axe2 = plt.subplot2grid((2,2),(1,0))
        axe2.set_ylabel("Heures d'inconfort (T>Tconf+2°C)")
        axe2.set_xlabel("Besoins de chauffage kWh/m2")
        plot2=axe2.scatter(x, z, c=y, alpha=0.5)
        axe2.scatter(economic_solution[0][0], economic_solution[0][1], marker="s",s=50, color="r")
        axe2.scatter(efficient_solution[0][0], efficient_solution[0][1], marker="s",s=50, color="k")
        axe2.scatter(comfortable_solution[0][0], comfortable_solution[0][1], marker="s",s=50, color="y")
        axe2.scatter(compromis_solution[0][0], compromis_solution[0][1], marker="s",s=50, color="g")
        axe2.scatter(utopia_solution[0][0],utopia_solution[0][1], marker="s",s=50, color="grey")
        #axe2.scatter(KS_solution[0][0], KS_solution[0][1], marker="s",s=50, color="b")
        plt.colorbar(plot2,ax=axe2,label="Cout actualisé en euros")

        axe3 = plt.subplot2grid((2,2),(1,1))
        axe3.set_xlabel("Cout actualisé en euros/m2")
        plot3 = axe3.scatter(y, z, c=x, alpha=0.5)
        axe3.scatter(economic_solution[0][2], economic_solution[0][1], marker="s", s=50,color="r")
        axe3.scatter(efficient_solution[0][2], efficient_solution[0][1], marker="s",s=50, color="k")
        axe3.scatter(comfortable_solution[0][2], comfortable_solution[0][1], marker="s",s=50, color="y")
        axe3.scatter(compromis_solution[0][2], compromis_solution[0][1], marker="s",s=50, color="g")
        axe3.scatter(utopia_solution[0][2],utopia_solution[0][1], marker="s",s=50, color="grey")
        #axe3.scatter(KS_solution[0][2], KS_solution[0][1], marker="s",s=50, color="b")
        plt.colorbar(plot3,ax=axe3,label="Besoins de chauffage kWh/m2")


        if base_solution != None: #.all()
            axe1.scatter(base_solution[0], base_solution[2], marker="s", s=50, color="purple", label="cas de référence")
            axe2.scatter(base_solution[0], base_solution[1], marker="s",s=50,  color="purple")
            axe3.scatter(base_solution[2], base_solution[1], marker="s", s=50, color="purple")
        if KS != None: #.all()
            axe1.scatter(KS[0], KS[2], marker="s", s=50, color="cyan", label="compromis_KS")
            axe2.scatter(KS[0], KS[1], marker="s",s=50,  color="cyan")
            axe3.scatter(KS[2], KS[1], marker="s", s=50, color="cyan")
        if not df2.empty:
            axe1.scatter(x2, y2, c='b', alpha=0.5)
            axe2.scatter(x2, z2, c='b', alpha=0.5)
            axe3.scatter(y2, z2, c='b', alpha=0.5)
        #fig.legend(loc="right")
        plt.savefig('./graphes/'+label+'_fronts.png')
        plt.show()
        
    if interactive == True:
        """To interactively analyze the results of the optimization"""
        import hvplot.pandas
        front=df.hvplot.scatter(x="besoins de chauffage kWh/m2", 
                                y="Cout global actualisé en euros/m2", 
                                c="heures d'inconfort (T>Tconf+2°C)",
                                clabel="heures d'inconfort (T>Tconf+2°C)",
                                hover_cols=["ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_fenetre"])
        hvplot.show(front)
        from bokeh.resources import INLINE
        hvplot.save(front, 'fronthvplot.html', resources=INLINE)

def comparaison_objectifs_deterministic_nomass(df_deterministic,  df_nomass,df_app, plot3D=False, label="front"):
    """trace les fonctions objectifs des deux fronts de pareto sur la même figure"""

    y_deterministic = df_deterministic["besoins de chauffage kWh/m2"]#chauffage
    x_deterministic = df_deterministic["Cout global actualisé en euros/m2"] #cout
    z_deterministic = df_deterministic["heures d'inconfort (T>Tconf+2°C)"] #inconfort

    y_nomass = df_nomass["besoins de chauffage kWh/m2"] #chauffage
    x_nomass = df_nomass["Cout global actualisé en euros/m2"] #cout
    z_nomass = df_nomass["heures d'inconfort (T>Tconf+2°C)"] #inconfort

    y_app = df_app["besoins de chauffage kWh/m2"] #chauffage
    x_app = df_app["Cout global actualisé en euros/m2"] #cout
    z_app = df_app["heures d'inconfort (T>Tconf+2°C)"] #inconfort


    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((2,2),(0,0))
    axe1.set_ylabel('Cout actualisé en euros/m2')
    axe1.scatter(y_deterministic, x_deterministic, c='b', alpha=0.5)# s=(type_fen_deter+1)*10)
    axe1.scatter(y_nomass, x_nomass,c='r', alpha=0.5)# s=(type_fen_nomass+1)*10)
    axe1.scatter(y_app, x_app, c='g', alpha=0.5)

    axe2 = plt.subplot2grid((2,2),(1,0))
    axe2.set_ylabel("Heures d'inconfort")
    axe2.set_xlabel("Besoins de chauffage kWh/m2")
    axe2.scatter(x_deterministic, z_deterministic, c='b',alpha=0.5)
    axe2.scatter(x_nomass, z_nomass,c='r', alpha=0.5)
    axe2.scatter(x_app, z_app,c='g', alpha=0.5)

    axe3 = plt.subplot2grid((2,2),(1,1))
    axe3.set_xlabel("Cout actualisé en euros/m2")
    axe3.scatter(y_deterministic, z_deterministic, c='b', alpha=0.5, label='Scénarii fixes')
    axe3.scatter(y_nomass, z_nomass,c='r', alpha=0.5, label='NoMASS') 
    axe3.scatter(y_app, z_app, c='g', alpha=0.5, label='NoMASS Approché')
    fig.legend()
    plt.savefig('./graphes/'+label+'.png')
    if plot3D == True:
        """to plot the tree functions"""
        fig = plt.figure()
        ax = plt.axes(projection='3d')#plt.subplot2grid((2,2),(0,1),projection='3d')
        ax.scatter(x_deterministic, y_deterministic, z_deterministic,color='b')#, c= 'b'
        ax.scatter(x_nomass, y_nomass, z_nomass,color='r')#, c= 'r'
        ax.scatter(x_app, y_app, z_app, c= 'g', label='NoMASS Approché')
        #ax.view_init(-140, 60)
        ax.set_ylabel("besoins de chauffage kWh/m2", size=8)
        ax.set_xlabel("Cout actualisé en euros/m2", size=8)
        ax.set_zlabel("heures d'inconfort (T>Tconf+2°C)", size=8)
        ax.set_xlim(min(x_deterministic),max(x_nomass))
        ax.figure.savefig('./graphes/'+label+'_3D.png')
    
    plt.show()
def comparaison_objectifs_nomass_approche(df_deterministic,  df_nomass, plot3D=False, label="front"):
    """trace les fonctions objectifs des deux fronts de pareto sur la même figure"""
    x_deterministic = df_deterministic["besoins de chauffage kWh/m2"]#chauffage
    y_deterministic = df_deterministic["Cout global actualisé en euros/m2"] #cout
    z_deterministic = df_deterministic["heures d'inconfort (T>Tconf+2°C)"] #inconfort

    x_nomass = df_nomass["besoins de chauffage kWh/m2"] #chauffage
    y_nomass = df_nomass["Cout global actualisé en euros/m2"] #cout
    z_nomass = df_nomass["heures d'inconfort (T>Tconf+2°C)"] #inconfort

    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((2,2),(0,0))
    axe1.set_ylabel('Cout actualisé en euros/m2')
    plot1=axe1.scatter(x_deterministic, y_deterministic, c='g', alpha=0.5)
    axe1.scatter(x_nomass, y_nomass,c='r', alpha=0.5)

    axe2 = plt.subplot2grid((2,2),(1,0))
    axe2.set_ylabel("Heures d'inconfort (T>Tconf+2°C)")
    axe2.set_xlabel("Besoins de chauffage kWh/m2")
    plot2=axe2.scatter(x_deterministic, z_deterministic, c='g',alpha=0.5)
    axe2.scatter(x_nomass, z_nomass,c='r', alpha=0.5)

    axe3 = plt.subplot2grid((2,2),(1,1))
    axe3.set_xlabel("Cout actualisé en euros/m2")
    plot3 = axe3.scatter(y_deterministic, z_deterministic, c='g', alpha=0.5, label='NoMASS Approché')
    axe3.scatter(y_nomass, z_nomass,c='r', alpha=0.5, label='NoMASS')
    fig.legend()
    if plot3D == True:
        """to plot the tree functions"""
        fig = plt.figure()
        ax = plt.axes(projection='3d')#plt.subplot2grid((2,2),(0,1),projection='3d')
        ax.scatter(x_deterministic, y_deterministic, z_deterministic,color='g')#, c= 'b'
        ax.scatter(x_nomass, y_nomass, z_nomass,color='r')#, c= 'r'
        ax.set_xlabel("besoins de chauffage kWh/m2", size=8)
        ax.set_ylabel("Cout actualisé en euros/m2", size=8)
        ax.set_zlabel("heures d'inconfort (T>Tconf+2°C)", size=8)
    plt.show()

def same_cost_min_heating (Pareto_objective_functions,Pareto_decision_parametres, base_solution):
    df = pd.DataFrame({"Besoins de chauffage kWh/m2" : Pareto_objective_functions[:, 0], 
                        "Heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions[:, 1],
                        "Cout global actualisé en euros" :  Pareto_objective_functions[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres[:, 2],
                        "type_fenetre" : Pareto_decision_parametres[:, 3]
                        })    
    index_cout=df[df["Cout global actualisé en euros/m2"]==base_solution[2]/97.5].index.values
    solution=df.iloc[index_cout]
    solution=solution.values.tolist()

def comparaison_param_chauff_deterministic_nomass(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon besoins de chauff"""

    chauff_deter = df_deterministic["besoins de chauffage kWh/m2"]#chauffage
    ep_murs_deter = df_deterministic["ep_murs_ext"] #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    type_fen_deter=df_deterministic["type_fenetre"]

    chauff_nomass = df_nomass["besoins de chauffage kWh/m2"] #
    ep_murs_nomass = df_nomass["ep_murs_ext"] #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"]
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    type_fen_nomass=df_nomass["type_fenetre"]

    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((4,2),(0,0))
    axe1.set_ylabel('ep_murs_ext')
    axe1.scatter(chauff_deter, ep_murs_deter, alpha=0.5, c='b')

    axe2 = plt.subplot2grid((4,2),(0,1))
    axe2.scatter(chauff_nomass, ep_murs_nomass, alpha=0.5, c='r')

    axe3 = plt.subplot2grid((4,2),(1,0))
    axe3.set_ylabel("ep_plancher_haut_cm")   
    axe3.scatter(chauff_deter, ep_phaut_deter,alpha=0.5, c='b')

    axe4 = plt.subplot2grid((4,2),(1,1))
    axe4.scatter(chauff_nomass, ep_phaut_nomass,alpha=0.5, c='r')

    axe5 = plt.subplot2grid((4,2),(2,0))
    axe5.set_ylabel("ep_plancher_bas_cm")  
    axe5.scatter(chauff_deter, ep_pbas_deter,alpha=0.5, c='b')

    axe6 = plt.subplot2grid((4,2),(2,1))
    axe6.scatter(chauff_nomass, ep_pbas_nomass,alpha=0.5, c='r')

    axe7 = plt.subplot2grid((4,2),(3,0))
    axe7.set_xlabel("Besoins de chauffage kWh/m2")
    axe7.set_ylabel("type_fen")  
    axe7.scatter(chauff_deter, type_fen_deter,alpha=0.5, c='b')

    axe8 = plt.subplot2grid((4,2),(3,1))
    axe8.set_xlabel("Besoins de chauffage kWh/m2")
    axe8.scatter(chauff_nomass, type_fen_nomass,alpha=0.5, c='r')
    fig.legend()
    plt.savefig('./graphes/param_chauff.png')
    #plt.show()
def comparaison_param_inconf_deterministic_nomass(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon inconf"""

    chauff_deter = df_deterministic["heures d'inconfort (T>Tconf+2°C)"]
    ep_murs_deter = df_deterministic["ep_murs_ext"] #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    type_fen_deter=df_deterministic["type_fenetre"]

    chauff_nomass = df_nomass["heures d'inconfort (T>Tconf+2°C)"] #
    ep_murs_nomass = df_nomass["ep_murs_ext"] #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"]
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    type_fen_nomass=df_nomass["type_fenetre"]

    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((4,2),(0,0))
    axe1.set_ylabel('ep_murs_ext')
    axe1.scatter(chauff_deter, ep_murs_deter, alpha=0.5, c='b')

    axe2 = plt.subplot2grid((4,2),(0,1))
    axe2.scatter(chauff_nomass, ep_murs_nomass, alpha=0.5, c='r')

    axe3 = plt.subplot2grid((4,2),(1,0))
    axe3.set_ylabel("ep_plancher_haut_cm")   
    axe3.scatter(chauff_deter, ep_phaut_deter,alpha=0.5, c='b')

    axe4 = plt.subplot2grid((4,2),(1,1))
    axe4.scatter(chauff_nomass, ep_phaut_nomass,alpha=0.5, c='r')

    axe5 = plt.subplot2grid((4,2),(2,0))
    axe5.set_ylabel("ep_plancher_bas_cm")  
    axe5.scatter(chauff_deter, ep_pbas_deter,alpha=0.5, c='b')

    axe6 = plt.subplot2grid((4,2),(2,1))
    axe6.scatter(chauff_nomass, ep_pbas_nomass,alpha=0.5, c='r')

    axe7 = plt.subplot2grid((4,2),(3,0))
    axe7.set_xlabel("heures d'inconfort")
    axe7.set_ylabel("type_fen")  
    axe7.scatter(chauff_deter, type_fen_deter,alpha=0.5, c='b')

    axe8 = plt.subplot2grid((4,2),(3,1))
    axe8.set_xlabel("heures d'inconfort")
    axe8.scatter(chauff_nomass, type_fen_nomass,alpha=0.5, c='r')
    fig.legend()
    plt.savefig('./graphes/param_inconf.png')
    #plt.show()
def comparaison_param_cout_deterministic_nomass(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon inconf"""

    chauff_deter = df_deterministic["Cout global actualisé en euros/m2"]
    ep_murs_deter = df_deterministic["ep_murs_ext"] #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    type_fen_deter=df_deterministic["type_fenetre"]

    chauff_nomass = df_nomass["Cout global actualisé en euros/m2"] #
    ep_murs_nomass = df_nomass["ep_murs_ext"] #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"]
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    type_fen_nomass=df_nomass["type_fenetre"]

    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((4,2),(0,0))
    axe1.set_ylabel('ep_murs_ext')
    axe1.scatter(chauff_deter, ep_murs_deter, alpha=0.5, c='b')

    axe2 = plt.subplot2grid((4,2),(0,1))
    axe2.scatter(chauff_nomass, ep_murs_nomass, alpha=0.5, c='r')

    axe3 = plt.subplot2grid((4,2),(1,0))
    axe3.set_ylabel("ep_plancher_haut_cm")   
    axe3.scatter(chauff_deter, ep_phaut_deter,alpha=0.5, c='b')

    axe4 = plt.subplot2grid((4,2),(1,1))
    axe4.scatter(chauff_nomass, ep_phaut_nomass,alpha=0.5, c='r')

    axe5 = plt.subplot2grid((4,2),(2,0))
    axe5.set_ylabel("ep_plancher_bas_cm")  
    axe5.scatter(chauff_deter, ep_pbas_deter,alpha=0.5, c='b')

    axe6 = plt.subplot2grid((4,2),(2,1))
    axe6.scatter(chauff_nomass, ep_pbas_nomass,alpha=0.5, c='r')

    axe7 = plt.subplot2grid((4,2),(3,0))
    axe7.set_xlabel("Cout actualisé en euros/m2")
    axe7.set_ylabel("type_fen")  
    axe7.scatter(chauff_deter, type_fen_deter,alpha=0.5, c='b')

    axe8 = plt.subplot2grid((4,2),(3,1))
    axe8.set_xlabel("Cout actualisé en euros/m2")
    axe8.scatter(chauff_nomass, type_fen_nomass,alpha=0.5, c='r')
    fig.legend()
    plt.savefig('./graphes/param_cout.png')
    plt.show()
def comparaison_param_cout_deterministic_nomass_norm(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon inconf"""

    cout_deter = df_deterministic["Cout global actualisé en euros/m2"]
    cout_deter_min=cout_deter.min()
    cout_deter_max=cout_deter.max()
    cout_deter_norm=(cout_deter-cout_deter_min)/(cout_deter_max-cout_deter_min)
    ep_murs_deter = df_deterministic["ep_murs_ext"] #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    type_fen_deter=df_deterministic["type_fenetre"]

    cout_nomass = df_nomass["Cout global actualisé en euros/m2"] #
    cout_nomass_min=cout_nomass.min()
    cout_nomass_max=cout_nomass.max()
    cout_nomass_norm=(cout_nomass-cout_nomass_min)/(cout_nomass_max-cout_nomass_min)
    ep_murs_nomass = df_nomass["ep_murs_ext"] #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"]
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    type_fen_nomass=df_nomass["type_fenetre"]

    fig = plt.figure()
    fig.set_size_inches(8,10)
        
    axe1 = plt.subplot2grid((4,1),(0,0))
    axe1.set_ylabel('ep_murs_ext')
    axe1.scatter(cout_deter_norm, ep_murs_deter, alpha=0.5, c='b')
    axe1.scatter(cout_nomass_norm, ep_murs_nomass, alpha=0.5, c='r')

    axe3 = plt.subplot2grid((4,1),(1,0))
    axe3.set_ylabel("ep_plancher_haut_cm")   
    axe3.scatter(cout_deter_norm, ep_phaut_deter,alpha=0.5, c='b')
    axe3.scatter(cout_nomass_norm, ep_phaut_nomass,alpha=0.5, c='r')

    axe5 = plt.subplot2grid((4,1),(2,0))
    axe5.set_ylabel("ep_plancher_bas_cm")  
    axe5.scatter(cout_deter_norm, ep_pbas_deter,alpha=0.5, c='b')
    axe5.scatter(cout_nomass_norm, ep_pbas_nomass,alpha=0.5, c='r')

    axe7 = plt.subplot2grid((4,1),(3,0))
    axe7.set_xlabel("Cout actualisé en euros/m2")
    axe7.set_ylabel("type_fen")  
    axe7.scatter(cout_deter_norm, type_fen_deter,alpha=0.5, c='b')
    axe7.scatter(cout_nomass_norm, type_fen_nomass,alpha=0.5, c='r')
    fig.legend()
    plt.savefig('./graphes/param_cout_norm.png')
    plt.show()
def comparaison_param_chauff_deterministic_nomass_norm(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon inconf"""

    chauff_deter = df_deterministic["besoins de chauffage kWh/m2"]
    chauff_deter_min=chauff_deter.min()
    chauff_deter_max=chauff_deter.max()
    chauff_deter_norm=(chauff_deter-chauff_deter_min)/(chauff_deter_max-chauff_deter_min)
    ep_murs_deter = df_deterministic["ep_murs_ext"] #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    type_fen_deter=df_deterministic["type_fenetre"]

    chauff_nomass = df_nomass["besoins de chauffage kWh/m2"] #
    chauff_nomass_min=chauff_nomass.min()
    chauff_nomass_max=chauff_nomass.max()
    chauff_nomass_norm=(chauff_nomass-chauff_nomass_min)/(chauff_nomass_max-chauff_nomass_min)
    ep_murs_nomass = df_nomass["ep_murs_ext"] #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"]
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    type_fen_nomass=df_nomass["type_fenetre"]

    fig = plt.figure()
    fig.set_size_inches(8,10)
        
    axe1 = plt.subplot2grid((4,1),(0,0))
    axe1.set_ylabel('ep_murs_ext')
    axe1.scatter(chauff_deter_norm, ep_murs_deter, alpha=0.5, c='b')
    axe1.scatter(chauff_nomass_norm, ep_murs_nomass, alpha=0.5, c='r')

    axe3 = plt.subplot2grid((4,1),(1,0))
    axe3.set_ylabel("ep_plancher_haut_cm")   
    axe3.scatter(chauff_deter_norm, ep_phaut_deter,alpha=0.5, c='b')
    axe3.scatter(chauff_nomass_norm, ep_phaut_nomass,alpha=0.5, c='r')

    axe5 = plt.subplot2grid((4,1),(2,0))
    axe5.set_ylabel("ep_plancher_bas_cm")  
    axe5.scatter(chauff_deter_norm, ep_pbas_deter,alpha=0.5, c='b')
    axe5.scatter(chauff_nomass_norm, ep_pbas_nomass,alpha=0.5, c='r')

    axe7 = plt.subplot2grid((4,1),(3,0))
    axe7.set_xlabel("besoins de chauffage kWh/m2")
    axe7.set_ylabel("type_fen")  
    axe7.scatter(chauff_deter_norm, type_fen_deter,alpha=0.5, c='b')
    axe7.scatter(chauff_nomass_norm, type_fen_nomass,alpha=0.5, c='r')
    fig.legend()
    plt.savefig('./graphes/param_chauff_nomr.png')
    plt.show()
def comparaison_param_inconf_deterministic_nomass_norm(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon inconf"""

    inconf_deter = df_deterministic["heures d'inconfort (T>Tconf+2°C)"]
    inconf_deter_min=inconf_deter.min()
    inconf_deter_max=inconf_deter.max()
    inconf_deter_norm=(inconf_deter-inconf_deter_min)/(inconf_deter_max-inconf_deter_min)
    ep_murs_deter = df_deterministic["ep_murs_ext"] #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    type_fen_deter=df_deterministic["type_fenetre"]

    inconf_nomass = df_nomass["heures d'inconfort (T>Tconf+2°C)"] #
    inconf_nomass_min=inconf_nomass.min()
    inconf_nomass_max=inconf_nomass.max()
    inconf_nomass_norm=(inconf_nomass-inconf_nomass_min)/(inconf_nomass_max-inconf_nomass_min)
    ep_murs_nomass = df_nomass["ep_murs_ext"] #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"]
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    type_fen_nomass=df_nomass["type_fenetre"]

    fig = plt.figure()
    fig.set_size_inches(8,10)
        
    axe1 = plt.subplot2grid((4,1),(0,0))
    axe1.set_ylabel('ep_murs_ext')
    axe1.scatter(inconf_deter_norm, ep_murs_deter, alpha=0.5, c='b')
    axe1.scatter(inconf_nomass_norm, ep_murs_nomass, alpha=0.5, c='r')

    axe3 = plt.subplot2grid((4,1),(1,0))
    axe3.set_ylabel("ep_plancher_haut_cm")   
    axe3.scatter(inconf_deter_norm, ep_phaut_deter,alpha=0.5, c='b')
    axe3.scatter(inconf_nomass_norm, ep_phaut_nomass,alpha=0.5, c='r')

    axe5 = plt.subplot2grid((4,1),(2,0))
    axe5.set_ylabel("ep_plancher_bas_cm")  
    axe5.scatter(inconf_deter_norm, ep_pbas_deter,alpha=0.5, c='b')
    axe5.scatter(inconf_nomass_norm, ep_pbas_nomass,alpha=0.5, c='r')

    axe7 = plt.subplot2grid((4,1),(3,0))
    axe7.set_xlabel("heures d'inconfort")
    axe7.set_ylabel("type_fen")  
    axe7.scatter(inconf_deter_norm, type_fen_deter,alpha=0.5, c='b')
    axe7.scatter(inconf_nomass_norm, type_fen_nomass,alpha=0.5, c='r')
    fig.legend()
    plt.savefig('./graphes/param_inconf_norm.png')
    plt.show()
def comparaison_param_norm_chauff_deterministic_nomass_norm(df_deterministic,  df_nomass):
    """distribution des paramètres optimaux selon chauff (tout est normalisé)"""

    chauff_deter = df_deterministic["besoins de chauffage kWh/m2"]
    chauff_deter_min=chauff_deter.min()
    chauff_deter_max=chauff_deter.max()
    chauff_deter_norm=(chauff_deter-chauff_deter_min)/(chauff_deter_max-chauff_deter_min)
    ep_murs_deter = df_deterministic["ep_murs_ext"]
    ep_murs_deter_min = ep_murs_deter.min()
    ep_murs_deter_max = ep_murs_deter.max()
    ep_murs_deter_norm=(ep_murs_deter-ep_murs_deter_min)/(ep_murs_deter_max-ep_murs_deter_min) #
    ep_phaut_deter = df_deterministic["ep_plancher_haut"] #
    ep_phaut_deter_min = ep_phaut_deter.min()
    ep_phaut_deter_max = ep_phaut_deter.max()
    ep_phaut_deter_norm=(ep_phaut_deter-ep_phaut_deter_min)/(ep_phaut_deter_max-ep_phaut_deter_min)
    ep_pbas_deter = df_deterministic["ep_plancher_bas"]
    ep_pbas_deter_min = ep_pbas_deter.min()
    ep_pbas_deter_max = ep_pbas_deter.max()
    ep_pbas_deter_norm=(ep_pbas_deter-ep_pbas_deter_min)/(ep_pbas_deter_max-ep_pbas_deter_min)
    type_fen_deter=df_deterministic["type_fenetre"]
    type_fen_deter.loc[type_fen_deter==0]=0.4
    type_fen_deter.loc[type_fen_deter==1]=0.7
    type_fen_deter.loc[type_fen_deter==2]=1.43
    type_fen_deter.loc[type_fen_deter==3]=2
    type_fen_deter_min = type_fen_deter.min()
    type_fen_deter_max = type_fen_deter.max()
    type_fen_deter_norm=(type_fen_deter-type_fen_deter_min)/(type_fen_deter_max-type_fen_deter_min)

    chauff_nomass = df_nomass["besoins de chauffage kWh/m2"] #
    chauff_nomass_min=chauff_nomass.min()
    chauff_nomass_max=chauff_nomass.max()
    chauff_nomass_norm=(chauff_nomass-chauff_nomass_min)/(chauff_nomass_max-chauff_nomass_min)
    ep_murs_nomass = df_nomass["ep_murs_ext"]
    ep_murs_nomass_min = ep_murs_nomass.min()
    ep_murs_nomass_max = ep_murs_nomass.max()
    ep_murs_nomass_norm=(ep_murs_nomass-ep_murs_nomass_min)/(ep_murs_nomass_max-ep_murs_nomass_min) #
    ep_phaut_nomass = df_nomass["ep_plancher_haut"] #
    ep_phaut_nomass_min = ep_phaut_nomass.min()
    ep_phaut_nomass_max = ep_phaut_nomass.max()
    ep_phaut_nomass_norm=(ep_phaut_nomass-ep_phaut_nomass_min)/(ep_phaut_nomass_max-ep_phaut_nomass_min)
    ep_pbas_nomass = df_nomass["ep_plancher_bas"]
    ep_pbas_nomass_min = ep_pbas_nomass.min()
    ep_pbas_nomass_max = ep_pbas_nomass.max()
    ep_pbas_nomass_norm=(ep_pbas_nomass-ep_pbas_nomass_min)/(ep_pbas_nomass_max-ep_pbas_nomass_min)
    type_fen_nomass=df_nomass["type_fenetre"]
    type_fen_nomass.loc[type_fen_nomass==0]=0.4
    type_fen_nomass.loc[type_fen_nomass==1]=0.7
    type_fen_nomass.loc[type_fen_nomass==2]=1.43
    type_fen_nomass.loc[type_fen_nomass==3]=2
    type_fen_nomass_min = type_fen_nomass.min()
    type_fen_nomass_max = type_fen_nomass.max()
    type_fen_nomass_norm=(type_fen_nomass-type_fen_nomass_min)/(type_fen_nomass_max-type_fen_deter_min)

    fig = plt.figure()
    fig.set_size_inches(8,10)
        
    axe1 = plt.subplot2grid((2,1),(0,0))
    axe1.scatter(chauff_deter_norm, ep_murs_deter_norm, alpha=0.5, c='b', label='ep_murs_ext')
    axe1.scatter(chauff_deter_norm, ep_phaut_deter_norm,alpha=0.5, c='k', label='ep_plancher_haut')
    axe1.scatter(chauff_deter_norm, ep_pbas_deter_norm,alpha=0.5, c='y', label='ep_plancher_bas')
    axe1.scatter(chauff_deter_norm, type_fen_deter_norm,alpha=0.5, c='g', label='type_vitrage')

    axe2 = plt.subplot2grid((2,1),(1,0))
    axe2.scatter(chauff_nomass_norm, ep_murs_nomass_norm, alpha=0.5, c='b')
    axe2.scatter(chauff_nomass_norm, ep_phaut_nomass_norm,alpha=0.5, c='k') 
    axe2.scatter(chauff_nomass_norm, ep_pbas_nomass_norm,alpha=0.5, c='y')
    axe2.scatter(chauff_nomass_norm, type_fen_nomass_norm,alpha=0.5, c='g')
    fig.legend()
    plt.savefig('./graphes/param_norm_chauff_norm.png')
    plt.show()
if __name__=="__main__":
#    Pareto_decision_parametres= np.array([ind for ind in population])
#    Pareto_objective_functions = np.array([ind.fitness.values for ind in population])
    
    #deterministic
    with open('./Results_To_Plot/pareto_obj_gen99_deter.csv', 'r') as f:
        Pareto_objective_functions_deterministic=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_objective_functions_deterministic=Pareto_objective_functions_deterministic.astype('float64')
    with open('./Results_To_Plot/pareto_param_gen99_deter.csv', 'r') as f:
        Pareto_decision_parametres_deterministic=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_decision_parametres_deterministic=Pareto_decision_parametres_deterministic.astype('float64')

    #nomass
    with open('./Results_To_Plot/pareto_obj_gen99_nomass.csv', 'r') as f:
        Pareto_objective_functions_nomass=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_objective_functions_nomass=Pareto_objective_functions_nomass.astype('float64')
    with open('./Results_To_Plot/pareto_param_gen99_nomass.csv', 'r') as f:
        Pareto_decision_parametres_nomass=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_decision_parametres_nomass=Pareto_decision_parametres_nomass.astype('float64')

    #nomass_coulpe_retraite
    with open('./Results_To_Plot/pareto_obj_gen99_nomass_retraite.csv', 'r') as f:
        Pareto_objective_functions_nomass_retraite=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_objective_functions_nomass_retraite=Pareto_objective_functions_nomass_retraite.astype('float64')
    with open('./Results_To_Plot/pareto_param_gen99_nomass_retraite.csv', 'r') as f:
        Pareto_decision_parametres_nomass_retraite=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_decision_parametres_nomass_retraite=Pareto_decision_parametres_nomass_retraite.astype('float64')
    #nomass_coulpe_jeune
    with open('./Results_To_Plot/pareto_obj_gen99_coupleActif.csv', 'r') as f:
        Pareto_objective_functions_nomass_jeune=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_objective_functions_nomass_jeune=Pareto_objective_functions_nomass_jeune.astype('float64')
    with open('./Results_To_Plot/pareto_param_gen99_coupleActif.csv', 'r') as f:
        Pareto_decision_parametres_nomass_jeune=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_decision_parametres_nomass_jeune=Pareto_decision_parametres_nomass_jeune.astype('float64')
    #mettre les resultats dans dataframe
    df_deterministic = pd.DataFrame({"besoins de chauffage kWh/m2" : Pareto_objective_functions_deterministic[:, 0], 
                        "heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions_deterministic[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions_deterministic[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres_deterministic[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres_deterministic[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres_deterministic[:, 2],
                        "type_fenetre" : Pareto_decision_parametres_deterministic[:, 3]
                        })
    df_nomass = pd.DataFrame({"besoins de chauffage kWh/m2" : Pareto_objective_functions_nomass[:, 0], 
                        "heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions_nomass[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions_nomass[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres_nomass[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres_nomass[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres_nomass[:, 2],
                        "type_fenetre" : Pareto_decision_parametres_nomass[:, 3]
                        })
'''    df_couple_retraite = pd.DataFrame({"besoins de chauffage kWh/m2" : Pareto_objective_functions_nomass_retraite[:, 0], 
                        "heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions_nomass_retraite[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions_nomass_retraite[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres_nomass_retraite[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres_nomass_retraite[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres_nomass_retraite[:, 2],
                        "type_fenetre" : Pareto_decision_parametres_nomass_retraite[:, 3]
                        })
    df_couple_jeune = pd.DataFrame({"besoins de chauffage kWh/m2" : Pareto_objective_functions_nomass_jeune[:, 0], 
                        "heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions_nomass_jeune[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions_nomass_jeune[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres_nomass_jeune[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres_nomass_jeune[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres_nomass_jeune[:, 2],
                        "type_fenetre" : Pareto_decision_parametres_nomass_jeune[:, 3]
                        })
'''                         
#df_deter_non_nomass= pd.read_excel("./Results_To_Plot/deter_non_nomass.xlsx", header=None, names=["ep_murs_ext","ep_plancher_haut","ep_plancher_bas",
                                                                                                           #"type_fenetre","besoins de chauffage kWh/m2","heures d'inconfort (T>Tconf+2°C)","Cout global actualisé en euros/m2"])
df_nomass_approche= pd.read_excel("./Results_To_Plot/nomass_approche.xlsx", header=None, names=["besoins de chauffage kWh/m2","heures d'inconfort (T>Tconf+2°C)","Cout global actualisé en euros/m2","ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_fenetre"])
df_avec_sur_all_comb=pd.read_excel("./Results_To_Plot/exhaustive_avec_surventilation_all_combinaisons.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
ref_solution_deter=[5.1, 147.35 , 158.48]
ref_solution_nomass=[11.51,135.59, 162.08]
KS_deter=[4.62,143.85,173.29]
KS_nomass=[9.53,155.37,173.65]
def plot_solutions_obj_bar():
    fig,axes=plt.subplots(nrows=3,ncols=3)
    axes[0,0].bar(["Ref","1","2","3","4","5"],[5.1,9.61,0.65,9.57,3.87,4.62],color=["purple","r","k","y","g","cyan"])#Scénarii fixes
    axes[0,1].bar(["Ref","1","2","3","4","5"],[11.51,20.94,3.98,16.35,13.04,10.55],color=["purple","r","k","y","g","cyan"])#Scénarii fixes simulé avec nomass
    axes[0,2].bar(["Ref","1","2","3","4","5"],[11.51,20.94,2.59,16.35,15.47,9.53],color=["purple","r","k","y","g","cyan"])#nomass

    axes[1,0].bar(["Ref","1","2","3","4","5"],[147.35,161.72,242.95,99.03,196.4,143.85],color=["purple","r","k","y","g","cyan"])#Scénarii fixes
    axes[1,1].bar(["Ref","1","2","3","4","5"],[135.59,122.9,223.44,103.71,168.28,137.85],color=["purple","r","k","y","g","cyan"])#Scénarii fixes simulé avec nomass
    axes[1,2].bar(["Ref","1","2","3","4","5"],[135.59,122.9,227.49,103.71,127.73,155.37],color=["purple","r","k","y","g","cyan"])#nomass
    
    axes[2,0].bar(["Ref","[10,10,10,3]","[50,30,50,0]","[10,10,10,0]","[20,35,10,3]","[15,25,10,0]"],[158.48,140.09,217.43,170.22,147.58,173.29],color=["purple","r","k","y","g","cyan"])#Scénarii fixes
    axes[2,1].bar(["Ref","[10,10,10,3]","[50,30,50,0]","[10,10,10,0]","[20,35,10,3]","[15,25,10,0]"],[162.08,146.43,217.09,174.02,145.49,176.61],color=["purple","r","k","y","g","cyan"])#Scénarii fixes simulé avec nomass
    axes[2,2].bar(["Ref","[10,10,10,3]","[50,40,50,0]","[10,10,10,0]","[15,20,10,3]","[40,35,10,2]"],[162.08,146.43,220.47,174.02,148.23,173.65],color=["purple","r","k","y","g","cyan"])#nomass
    
    axes[0,0].set_ylabel("Besoins de chauffage [kWh/m2]",fontsize=8)
    axes[1,0].set_ylabel("Nombre d'heures d'inconfort",fontsize=8)
    axes[2,0].set_ylabel("Coût actualisé [euros/m2]",fontsize=8)
    axes[0,0].set_title("Scénarii fixes",loc="left")
    axes[0,1].set_title("Scénarii fixes simulés avec NoMASS",loc="left")
    axes[0,2].set_title("NoMASS",loc="left")
    #50,30,50,0 3,97948718	223,4368205	217,091285
    #20,35,10,3 13,04215385	168,2763077	145,4880186
    #15,25,10,0 10,55	137,85	176,61
    axes[2,0].tick_params(labelrotation=90,labelsize=8)
    axes[2,1].tick_params(labelrotation=90,labelsize=8)
    axes[2,2].tick_params(labelrotation=90,labelsize=8)
    plt.show()
def plot_solutions_param_bar():
    fig,axes=plt.subplots(nrows=1,ncols=4)
    ep_murs=[["1",10,10],
      ["2",50,50],
      ["3",10,10],
      ["4",20,15],
      ["5",15,40],
      ["6",10,10]
    ]
    ep_ph=[["1",10,10],
      ["2",30,40],
      ["3",10,10],
      ["4",35,20],
      ["5",25,35],
      ["6",10,15]
    ]
    ep_pb=[["1",10,10],
      ["2",50,50],
      ["3",10,10],
      ["4",10,10],
      ["5",10,10],
      ["6",10,10]
    ]
    type_v=[["1",2,2],
      ["2",0.4,0.4],
      ["3",0.4,0.4],
      ["4",2,2],
      ["5",0.4,1.43],
      ["6",0.4,1.43]
    ]
    df_ep_murs=pd.DataFrame(ep_murs,columns=["Solution","Scénarii fixes","NoMASS"])
    df_ep_murs.plot(x="Solution", y=["Scénarii fixes","NoMASS"], kind="bar",ax=axes[0],color=["b","r"])
    df_ep_ph=pd.DataFrame(ep_ph,columns=["Solution","Scénarii fixes","NoMASS"])
    df_ep_ph.plot(x="Solution", y=["Scénarii fixes","NoMASS"], kind="bar",ax=axes[1],color=["b","r"])
    df_ep_pb=pd.DataFrame(ep_pb,columns=["Solution","Scénarii fixes","NoMASS"])
    df_ep_pb.plot(x="Solution", y=["Scénarii fixes","NoMASS"], kind="bar",ax=axes[2],color=["b","r"])
    df_type_v=pd.DataFrame(type_v,columns=["Solution","Scénarii fixes","NoMASS"])
    df_type_v.plot(x="Solution", y=["Scénarii fixes","NoMASS"], kind="bar",ax=axes[3],color=["b","r"])
    axes[0].set_ylabel("ep_murs_ext [cm]")
    axes[1].set_ylabel("ep_plancher_haut [cm]")
    axes[2].set_ylabel("ep_plancher_bas [cm]")
    axes[3].set_ylabel("Uw_vitrage [W/(m2.K)]")
    axes[0].set_yticks([10,20,30,40,50])
    axes[1].set_yticks([10,20,30,40,50])
    axes[2].set_yticks([10,20,30,40,50])
    '''axes[0].bar(["Ref","1","2","3","4","5"],[10,50,10,20,15,10],color="b",alpha=0.5, label="Scénarii fixes")#Scénarii fixes ["b","b","b","b","b","b"]
    axes[0].bar(["Ref","1","2","3","4","5"],[10,20,3,16,13,10],color="r", label="NoMASS")#nomass
    axes[1].bar(["Ref","1","2","3","4","5"],[10,50,10,20,15,10],color="b",alpha=0.5, label="Scénarii fixes")#Scénarii fixes
    axes[1].bar(["Ref","1","2","3","4","5"],[10,20,3,16,13,10],color="r", label="NoMASS")#
    axes[2].bar(["Ref","1","2","3","4","5"],[10,50,10,20,15,10],color="b",alpha=0.5, label="Scénarii fixes")#Scénarii fixes
    axes[2].bar(["Ref","1","2","3","4","5"],[10,20,3,16,13,10],color="r", label="NoMASS")#
    axes[3].bar(["Ref","1","2","3","4","5"],[10,50,10,20,15,10],color="b",alpha=0.5, label="Scénarii fixes")#Scénarii fixes
    axes[3].bar(["Ref","1","2","3","4","5"],[10,20,3,16,13,10],color="r", label="NoMASS")#
    fig.legend()'''
    plt.show()
#same_cost_min_heating (Pareto_objective_functions,Pareto_decision_parametres, ref_solution)
#print("la solution ayant le même cout mais moins de chauffage est\n", solution)

#plots(df_nomass, base_solution=ref_solution_nomass, KS=KS_nomass,plot2D = True, plot3D = True, interactive = False, label="nomass")
#plots(df_couple_retraite, base_solution=ref_solution_retraite, plot2D = True, plot3D = True, interactive = False, label="couple_retraite")
#plots(df_couple_jeune, base_solution=ref_solution_jeune, plot2D = True, plot3D = True, interactive = False, label="couple_actif")

comparaison_objectifs_deterministic_nomass(df_deterministic,  df_nomass,df_nomass_approche, label="comparaison deter nomass app", plot3D=True)
#plots(df_deterministic, base_solution=ref_solution_deter, KS=KS_deter, plot2D = True, plot3D = True, interactive = False, label="deter")
#comparaison_objectifs_jeune_retraite(df_couple_jeune,  df_couple_retraite, label="comparaison jeune retraite")
#comparaison_param_inconf_deterministic_nomass_norm(df_deterministic,  df_nomass)
#comparaison_param_inconf_deterministic_nomass(df_deterministic,  df_nomass)
x=1/3
y=1/3
z=1-x-y
'''solution_intermediaire_deter=find_solution(df_deterministic, x, y, z)
solution_intermediaire_nomass=find_solution(df_nomass, x, y, z)
solution_intermediaire_nomass_retraite=find_solution(df_couple_retraite, x, y, z)
solution_intermediaire_nomass_jeune=find_solution(df_couple_jeune, x, y, z)'''
#print(sort_solutions(df_deterministic, x, y, z))
#print(sort_solutions(df_nomass, x, y, z))
#print(sort_solutions(df_couple_retraite, x, y, z))
#print(sort_solutions(df_couple_jeune, x, y, z))
#print(find_solution_KS(df_deterministic))
#print (df_deter_non_nomass.head())
#comparaison_objectifs_deterministic_nomass(df_deterministic,  df_nomass, label="comparaison deter_non_nomass nomass", plot3D=True)
#comparaison_objectifs_nomass_approche(df_nomass_approche,  df_nomass, label="comparaison deter_non_nomass nomass", plot3D=True)
#comparaison_param_norm_chauff_deterministic_nomass_norm(df_deterministic,  df_nomass)
#print(df_nomass.groupby(["type_fenetre"]).min()[["besoins de chauffage kWh/m2","heures d'inconfort (T>Tconf+2°C)","Cout global actualisé en euros/m2"]])#[["besoins de chauffage kWh/m2","heures d'inconfort (T>Tconf+2°C)"]]
#hierarchie=df_deterministic.groupby(["type_fenetre","ep_murs_ext","ep_plancher_haut","ep_plancher_bas"]).count()
#hierarchie.to_excel("hierarchie.xlsx")
#df_nomass[["ep_murs_ext","ep_plancher_haut","ep_plancher_bas"]].hist(bins=[10,15,20,25,30,35,40,45,50,55],xticks=[10,15,20,25,30,35,40,45,50,55])
df_nomass.columns=["besoins chauffage","inconfort","cout actualisé","ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_vitrage"]
df_deterministic.columns=["besoins chauffage","inconfort","cout actualisé","ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_vitrage"]
df_nomass_approche.columns=["besoins chauffage","inconfort","cout actualisé","ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_vitrage"]
def plot_hist_param(df,df2):
    fig,axes=plt.subplots(nrows=2,ncols=2)
    b=[10,15,20,25,30,35,40,45,50,55]
    axes[0,0].hist([df["ep_murs_ext"],df2["ep_murs_ext"]],bins=b)#
    axes[1,0].hist([df["ep_plancher_haut"],df2["ep_plancher_haut"]],bins=b)
    axes[0,1].hist([df["ep_plancher_bas"],df2["ep_plancher_bas"]],bins=b)
    axes[1,1].hist([df["type_vitrage"],df2["type_vitrage"]], bins=[0,1,2,3,4])
    #plt.xticks(ticks=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
    fig.legend()
    axes[0,0].set_title("ep_murs_ext")
    axes[1,0].set_title("ep_plancher_haut")
    axes[0,1].set_title("ep_plancher_bas")
    axes[1,1].set_title("type_vitrage")
    plt.show()
def plot_hist_param2(df,df2):
    fig,axes=plt.subplots(nrows=2,ncols=4)
    b=[10,15,20,25,30,35,40,45,50,55]
    axes[0,0].hist([df["ep_murs_ext"]],bins=b, color='b', label="Scénarii fixes")
    axes[0,1].hist([df["ep_plancher_haut"]],bins=b, color='b')
    axes[0,2].hist([df["ep_plancher_bas"]],bins=b, color='b')
    axes[0,3].hist([df["type_vitrage"]], bins=[0,1,2,3,4], color='b')
    axes[1,0].hist([df2["ep_murs_ext"]],bins=b, color='r', label="NoMASS")
    axes[1,1].hist([df2["ep_plancher_haut"]],bins=b, color='r')
    axes[1,2].hist([df2["ep_plancher_bas"]],bins=b, color='r')
    axes[1,3].hist([df2["type_vitrage"]], bins=[0,1,2,3,4], color='r')
    fig.legend()
    axes[0,0].set_title("ep_murs_ext")
    axes[0,1].set_title("ep_plancher_haut")
    axes[0,2].set_title("ep_plancher_bas")
    axes[0,3].set_title("type_vitrage")
    plt.show()
def plot_hist_param3(df,df2):
    fig,axes=plt.subplots(nrows=1,ncols=4)
    m=df.shape[0]
    m2=df2.shape[0]
    b=[10,15,20,25,30,35,40,45,50,55]
    n0,x0,_=axes[0].hist([df["ep_murs_ext"]],bins=b, color='b', label="Scénarii fixes", alpha=0.5)
    n1,x1,_=axes[1].hist([df["ep_plancher_haut"]],bins=b, color='b', alpha=0.5)
    n2,x2,_=axes[2].hist([df["ep_plancher_bas"]],bins=b, color='b', alpha=0.5)
    n3,x3,_=axes[3].hist([df["type_vitrage"]], bins=[0,1,2,3,4], color='b', alpha=0.5)
    n02,x02,_=axes[0].hist([df2["ep_murs_ext"]],bins=b, color='r', label="NoMASS", alpha=0.5)
    n12,x12,_=axes[1].hist([df2["ep_plancher_haut"]],bins=b, color='r', alpha=0.5)
    n22,x22,_=axes[2].hist([df2["ep_plancher_bas"]],bins=b, color='r', alpha=0.5)
    n32,x32,_=axes[3].hist([df2["type_vitrage"]], bins=[0,1,2,3,4], color='r', alpha=0.5)
    fig,axes=plt.subplots(nrows=1,ncols=4,sharey=True)
    bin_centers_0=0.5*(x0[1:]+x0[:-1])
    axes[0].plot(bin_centers_0,n0*100/m, color='b', label="Scénarii fixes")
    bin_centers_02=0.5*(x02[1:]+x02[:-1])
    axes[0].plot(bin_centers_02,n02*100/m2, color='r', label="NoMASS")
    bin_centers_1=0.5*(x1[1:]+x1[:-1])
    axes[1].plot(bin_centers_1,n1*100/m, color='b')
    bin_centers_12=0.5*(x12[1:]+x12[:-1])
    axes[1].plot(bin_centers_12,n12*100/m2, color='r')
    bin_centers_2=0.5*(x2[1:]+x2[:-1])
    axes[2].plot(bin_centers_2,n2*100/m, color='b')
    bin_centers_22=0.5*(x22[1:]+x22[:-1])
    axes[2].plot(bin_centers_22,n22*100/m2, color='r')
    bin_centers_3=0.5*(x3[1:]+x3[:-1])
    axes[3].plot(bin_centers_3,n3*100/m, color='b')
    bin_centers_32=0.5*(x32[1:]+x32[:-1])
    axes[3].plot(bin_centers_32,n32*100/m2, color='r')
    fig.legend()
    axes[0].set_xlabel("ep_murs_ext [cm]")
    axes[1].set_xlabel("ep_plancher_haut [cm]")
    axes[2].set_xlabel("ep_plancher_bas [cm]")
    axes[3].set_xlabel("type_vitrage")
    axes[0].set_ylabel("% solutions")
    plt.show()
def plot_hist_obj(df,df2):
    fig,axes=plt.subplots(nrows=1,ncols=3)
    axes[0].hist([df["besoins chauffage"],df2["besoins chauffage"]], color=['b','r'], label=["Scénarii fixes",'NoMASS'])
    axes[1].hist([df["inconfort"],df2["inconfort"]], color=['b','r'])
    axes[2].hist([df["cout actualisé"],df2["cout actualisé"]],color=['b','r'])
    fig.legend()
    axes[0].set_title("Besoins de chauffage kWh/m2")
    axes[1].set_title("Nombre d'heures d'inconfort")
    axes[2].set_title("Cout actualisé en euros/m2")
    plt.show()
def plot_hist_obj2(df,df2):
    fig,axes=plt.subplots(nrows=1,ncols=3)
    axes[0].hist([df["besoins chauffage"]], alpha=0.5, color='b', label="Scénarii fixes")
    axes[1].hist([df["inconfort"]], color='b', alpha=0.5)
    axes[2].hist([df["cout actualisé"]],color='b', alpha=0.5)
    axes[0].hist([df2["besoins chauffage"]],color='r', label="NoMASS", alpha=0.5)
    axes[1].hist([df2["inconfort"]],color='r', alpha=0.5)
    axes[2].hist([df2["cout actualisé"]],color='r', alpha=0.5)
    fig.legend()
    axes[0].set_title("Besoins de chauffage kWh/m2")
    axes[1].set_title("Nombre d'heures d'inconfort")
    axes[2].set_title("Cout actualisé en euros/m2")
    plt.show()
def plot_hist_obj3(df,df2,df3):
    fig,axes=plt.subplots(nrows=1,ncols=3,sharey=True)
    m=df.shape[0]
    m2=df2.shape[0]
    m3=df3.shape[0]
    n0,x0,_=axes[0].hist([df["besoins chauffage"]], alpha=0.5, color='b', label="Scénarii fixes")
    n1,x1,_=axes[1].hist([df["inconfort"]], color='b', alpha=0.5)
    n2,x2,_=axes[2].hist([df["cout actualisé"]],color='b', alpha=0.5)
    n02,x02,_=axes[0].hist([df2["besoins chauffage"]],color='r', label="NoMASS", alpha=0.5)
    n12,x12,_=axes[1].hist([df2["inconfort"]],color='r', alpha=0.5)
    n22,x22,_=axes[2].hist([df2["cout actualisé"]],color='r', alpha=0.5)
    n03,x03,_=axes[0].hist([df3["besoins chauffage"]],color='g', label="NoMASS Approché", alpha=0.5)
    n13,x13,_=axes[1].hist([df3["inconfort"]],color='g', alpha=0.5)
    n23,x23,_=axes[2].hist([df3["cout actualisé"]],color='g', alpha=0.5)
    fig.legend()
    axes[0].set_title("Besoins de chauffage kWh/m2")
    axes[1].set_title("Nombre d'heures d'inconfort")
    axes[2].set_title("Cout actualisé en euros/m2")
    fig,axes=plt.subplots(nrows=1,ncols=3,sharey=True)
    bin_centers_0=0.5*(x0[1:]+x0[:-1])
    axes[0].plot(bin_centers_0,n0*100/m, color='b', label="Scénarii fixes")
    bin_centers_02=0.5*(x02[1:]+x02[:-1])
    axes[0].plot(bin_centers_02,n02*100/m2, color='r', label="NoMASS")
    bin_centers_03=0.5*(x03[1:]+x03[:-1])
    axes[0].plot(bin_centers_03,n03*100/m3, color='g', label="NoMASS Approché")
    bin_centers_1=0.5*(x1[1:]+x1[:-1])
    axes[1].plot(bin_centers_1,n1*100/m, color='b')
    bin_centers_12=0.5*(x12[1:]+x12[:-1])
    axes[1].plot(bin_centers_12,n12*100/m2, color='r')
    bin_centers_13=0.5*(x13[1:]+x13[:-1])
    axes[1].plot(bin_centers_13,n13*100/m3, color='g')
    bin_centers_2=0.5*(x2[1:]+x2[:-1])
    axes[2].plot(bin_centers_2,n2*100/m, color='b')
    bin_centers_22=0.5*(x22[1:]+x22[:-1])
    axes[2].plot(bin_centers_22,n22*100/m2, color='r')
    bin_centers_23=0.5*(x23[1:]+x23[:-1])
    axes[2].plot(bin_centers_23,n23*100/m3, color='g')
    fig.legend()
    axes[0].set_xlabel("Besoins de chauffage [kWh/m2]")
    axes[1].set_xlabel("Nombre d'heures d'inconfort")
    axes[2].set_xlabel("Cout actualisé [euros/m2]")
    axes[0].set_ylabel("% solutions")
    plt.show()
def calcul_hypervolume(pareto):
    """
    Plots the hypervolume of the population to monitor the optimization

    Args:
        pareto (pareto): contains population of Pareto front
        df: population

    Returns:

    """
    pareto.columns=["besoins chauffage","inconfort","cout actualisé","ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_vitrage"]
    #df_deterministic.columns=["besoins chauffage","inconfort","cout actualisé","ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_vitrage"]

    print("calcul hypervolume")
    try:
        # try importing the C version
        from deap.tools._hypervolume import hv
    except ImportError:
        # fallback on python version
        from deap.tools._hypervolume import pyhv as hv
    pareto=pareto[["besoins chauffage","inconfort","cout actualisé"]]
    wobj = pareto.to_numpy()
    ref = np.array([20.93774359,228.37394872,220.46708661])# point nadir nomass
    #print(np.max(wobj, axis=0))# * 1.5   # Pose problème np.array([100,100,100])
    hypervolume=hv.hypervolume(wobj, ref)
    print("calculation of HV done")
    print(ref,hypervolume)
    return hypervolume, wobj
#calcul_hypervolume(df_nomass) 
#print(df_nomass_approche)
#plot_hist_obj3(df_deterministic,df_nomass,df_nomass_approche)
#plot_hist_param3(df_deterministic,df_nomass)
#df_deterministic.groupby(['type_vitrage']).size().unstack().plot(kind='bar', stacked=True)
#plt.show()
#plots_3d_grouped(df_nomass)
#plots_grouped(df_nomass)
#plots_2d_grouped(df_deterministic)
#plots_2d_grouped(df_nomass)

def boxplot_pareto(df,df2,df3):
    fig,axes=plt.subplots(nrows=1,ncols=3)
    box=axes[0].boxplot([df["besoins chauffage"],df2["besoins chauffage"],df3["besoins chauffage"]],labels=["Scénarii fixes","NoMASS","NoMASS Approché"])
    axes[1].boxplot([df["inconfort"],df2["inconfort"],df3["inconfort"]],labels=["Scénarii fixes","NoMASS","NoMASS Approché"])
    axes[2].boxplot([df["cout actualisé"],df2["cout actualisé"],df3["cout actualisé"]],labels=["Scénarii fixes","NoMASS","NoMASS Approché"])
    axes[0].set_title('Besoins de chauffage [kWh/m2]')
    axes[1].set_title("Nombre d'heures d'inconfort")
    axes[2].set_title("Coût actualisé [euros/m2]")
    plt.show()
#boxplot_pareto(df_deterministic, df_nomass,df_nomass_approche)
'''print(find_best_solution(df_deterministic)) #sans normalisation
print(find_solution_KS(df_deterministic)) #avec normalisation
print(find_solution_utopia(df_deterministic)) #sans normalisation'''
#plot_solutions_param_bar()
def plot_exhaustive(df1,df2):
    x1 = df1["f1"]#chauffage
    y1 = df1["f2"] #cout
    z1 = df1["f3"] #inconfort
    x2 = df2["f1"]#chauffage
    y2 = df2["f2"] #cout
    z2 = df2["f3"] #inconfort
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x1, y1, z1, alpha=0.5,color='gray')
    ax.scatter(x2, y2, z2, alpha=0.5,color='r')
    plt.show()
    fig = plt.figure()   
    axe1 = plt.subplot2grid((2,2),(0,0))
    axe1.set_ylabel('Cout actualisé en euros/m2')
    axe1.scatter(x1, y1, c="gray", alpha=0.5,label="tous les individus")
    axe1.scatter(x2, y2, c="r", alpha=0.5, label="solutions non-dominées")
    axe2 = plt.subplot2grid((2,2),(1,0))
    axe2.set_ylabel("Heures d'inconfort (T>Tconf+2°C)")
    axe2.set_xlabel("Besoins de chauffage kWh/m2")
    axe2.scatter(x1, z1, c="gray", alpha=0.5)
    axe2.scatter(x2, z2, c="r", alpha=0.5)
    axe3 = plt.subplot2grid((2,2),(1,1))
    axe3.set_xlabel("Cout actualisé en euros/m2")
    axe3.scatter(y1, z1, c='gray', alpha=0.5)
    axe3.scatter(y2, z2, c='r', alpha=0.5)
    fig.legend()
    plt.show()
'''df_avec_sur_all_comb=pd.read_excel("./Results_To_Plot/exhaustive_avec_surventilation_all_combinaisons.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
df_avec_sur_pareto=pd.read_excel("exhaustive_avec_surventilation_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
df_sans_sur_all_comb=pd.read_excel("./Results_To_Plot/exhaustive_sans_surventilation_all_combinaisons.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
df_sans_sur_pareto=pd.read_excel("exhaustive_sans_surventilation_pareto.xlsx", header=None, names=["f1","f2","f3","x1","x2","x3","x4"])
plot_exhaustive(df_sans_sur_all_comb,df_sans_sur_pareto)'''
