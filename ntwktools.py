import numpy as np
import iberoSignalPro.preprocesa as ib
import matplotlib.pyplot as plt
import mne
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
import networkx as nx

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import contextlib
import os

import logging

logging.basicConfig(level=logging.CRITICAL)


from scipy.stats import kruskal
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel


def compute_kruskal_wallis_anova(df, group_column, value_column):
    """
    Computes the Kruskal-Wallis H-test for given groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare across groups.

    Returns:
    - The test statistic and p-value from the Kruskal-Wallis H-test.
    """
    
    groups = df[group_column].unique()
    
    group_data = [df[df[group_column] == group][value_column] for group in groups]
    
    test_stat, p_value = kruskal(*group_data)
    
    return test_stat, p_value


def compute_one_way_anova(df, group_column, value_column):
    """
    Computes the one-way ANOVA test for given groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare across groups.

    Returns:
    - The F statistic and p-value from the one-way ANOVA test.
    """
    
    groups = df[group_column].unique()
    
    group_data = [df[df[group_column] == group][value_column] for group in groups]
    
    F_stat, p_value = f_oneway(*group_data)
    
    return F_stat, p_value


def compute_two_way_anova(df, dependent_var, factor1, factor2):
    """
    Computes the two-way ANOVA for a given DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - dependent_var: The name of the dependent variable (continuous).
    - factor1: The name of the first factor (independent variable).
    - factor2: The name of the second factor (independent variable).

    Returns:
    - ANOVA table as a DataFrame.
    """
    # Construct the formula for the two-way ANOVA
    formula = f'{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'
    
    # Fit the model
    model = ols(formula, data=df).fit()
    
    # Perform ANOVA and return the table
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def compute_t_test(df, group_column, value_column):
    """
    Computes the t-test for the means of two independent groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare.
    - group1_label: The label of the first group for comparison.
    - group2_label: The label of the second group for comparison.

    Returns:
    - The T statistic and p-value from the t-test.
    """
    groups = df[group_column].unique()
    if(groups.shape[0] != 2):
        raise ValueError("Los gurpos deben de ser unicamente dos")
    # Extract data for the two specified groups
    group1_data = df[df[group_column] == groups[0]][value_column]
    group2_data = df[df[group_column] == groups[1]][value_column]
    
    # Perform the t-test
    T_stat, p_value = ttest_ind(group1_data, group2_data)
    
    return T_stat, p_value



def compute_paired_t_test(df, group_column, value_column):
    """
    Computes the paired t-test for the means of two related groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare.

    Returns:
    - The T statistic and p-value from the paired t-test.
    """
    groups = df[group_column].unique()
    if len(groups) != 2:
        raise ValueError("Los gurpos deben de ser unicamente dos")
    
    # Extract data for the two specified groups
    group1_data = df[df[group_column] == groups[0]][value_column]
    group2_data = df[df[group_column] == groups[1]][value_column]
    
    # Ensure both groups have the same number of observations
    if len(group1_data) != len(group2_data):
        raise ValueError("Both groups must have the same number of observations for a paired t-test.")
    
    # Perform the paired t-test
    T_stat, p_value = ttest_rel(group1_data, group2_data)
    
    return T_stat, p_value

def get_significat_cols(df_analisis, p_val = 0.01, label = "label", test = "paired_t_test", verbose = False):
    significat_cols = []
    p_values = []
    for col in df_analisis.columns:
        if col == label:
            continue
        if test == "paired_t_test":
            T_stat, p_value = compute_paired_t_test(df_analisis, label, col)
        elif test == "t_test":
            T_stat, p_value = compute_t_test(df_analisis,label, col)
        elif test == "kruskal_wallis":
            T_stat, p_value = compute_kruskal_wallis_anova(df_analisis,label, col)
        elif test == "one_way_anova":
            T_stat, p_value = compute_one_way_anova(df_analisis, label, col)
        if p_value < p_val:
            if verbose:
                print(f" {col} - {p_value}")
            p_values.append(p_value)
            significat_cols.append(col)
    return significat_cols, p_values

# Seleccion de ventanas buenas

def selecciona_ventanas(act, rep, fs = 10, win_len = 15, siPlot = False, signal = None):
    '''
    Selecciona las ventanas de actividad y reposo que tengan una duración mayor o igual a win_len
    act: lista con los indices de las ventanas de actividad
    rep: lista con los indices de las ventanas de reposo
    fs: frecuencia de muestreo
    win_len: duración mínima de las ventanas en segundos
    siPlot: si se quiere graficar las ventanas seleccionadas
    signal: señal de la cual se obtuvieron las ventanas
    '''
    new_act = []
    new_rep = []
    win_len = win_len * fs
    for index_act, index_rep in zip(act, rep):
        start_act, end_act = index_act
        start_rep, end_rep = index_rep

        if end_act - start_act >= win_len:
            start_act = start_act + (end_act - start_act - win_len) // 2
            end_act = start_act + win_len
            new_act.append((start_act, end_act))
            if siPlot:
                if signal is None:
                    raise ValueError("signal is needed for plotting")
                plt.plot(signal[start_act:end_act])
                

        if end_rep - start_rep >= win_len:
            start_rep = start_rep + (end_rep - start_rep - win_len) // 2
            end_rep = start_rep + win_len
            new_rep.append((start_rep, end_rep))
            if siPlot:
                if signal is None:
                    raise ValueError("signal is needed for plotting")
                plt.plot(signal[start_rep:end_rep])
        
        
    return new_act, new_rep

def obtener_win(sig, binary_sig, siPlot=True):
        # Ensure binary_sig is binary
        binary_sig = np.array(binary_sig).flatten()
        binary_sig = (binary_sig >= 0.5).astype(int)  # Convert to binary (0 or 1)

        diff = np.diff(binary_sig)
        idx_actividad = np.where(diff == 1)[0]
        idx_rep = np.where(diff == -1)[0]

        print(binary_sig.shape)

        if binary_sig[0] == 1:
            idx_actividad = np.insert(idx_actividad, 0, 0)
        #if binary_sig[-1] == 1:
        #    idx_actividad = np.append(idx_actividad, len(binary_sig) -1)
        
        if binary_sig[0] == 0:
            idx_rep = np.insert(idx_rep, 0, 0)
        #if binary_sig[-1] == 0:
        #    idx_rep = np.append(idx_rep, len(binary_sig) - 1)
        
        #print(idx_actividad)
        #print(idx_rep)
        
        # Ensure idx_actividad and idx_rep have the same length by adding samples
        while len(idx_actividad) < len(idx_rep):
            idx_actividad = np.append(idx_actividad, idx_actividad[-1])
        while len(idx_rep) < len(idx_actividad):
            idx_rep = np.append(idx_rep, idx_rep[-1])
        
        if idx_rep[0] < idx_actividad[0]:
            ventanas_reposo = np.stack((idx_rep, idx_actividad)).T
            ventanas_actividad = np.stack((idx_actividad[:-1], idx_rep[1:])).T
        else:
            ventanas_reposo = np.stack((idx_rep[:-1], idx_actividad[1:])).T
            ventanas_actividad = np.stack((idx_actividad[:], idx_rep[:])).T

        #print(ventanas_reposo.shape)
        #print(ventanas_actividad.shape)

        if siPlot:
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 2, 1) 
            for ventana in ventanas_actividad:
                plt.plot(sig[ventana[0]: ventana[1]])
            plt.title('Ventanas de actividad')

            plt.subplot(1, 2, 2)  
            for ventana in ventanas_reposo:
                plt.plot(sig[ventana[0]: ventana[1]])
            plt.title('Ventanas de reposo')

            plt.show()

        return ventanas_actividad, ventanas_reposo


def selecciona_ventanas(act, rep, fs = 10, win_len = 20, siPlot = False, signal = None):
    '''
    Selecciona las ventanas de actividad y reposo que tengan una duración mayor o igual a win_len
    act: lista con los indices de las ventanas de actividad
    rep: lista con los indices de las ventanas de reposo
    fs: frecuencia de muestreo
    win_len: duración mínima de las ventanas en segundos
    siPlot: si se quiere graficar las ventanas seleccionadas
    signal: señal de la cual se obtuvieron las ventanas
    '''
    new_act = []
    new_rep = []
    win_len = win_len * fs
    for index_act, index_rep in zip(act, rep):
        start_act, end_act = index_act
        start_rep, end_rep = index_rep

        if end_act - start_act >= win_len:
            start_act = start_act + (end_act - start_act - win_len) // 2
            end_act = start_act + win_len
            new_act.append((start_act, end_act))
            if siPlot:
                if signal is None:
                    raise ValueError("signal is needed for plotting")
                plt.plot(signal[start_act:end_act])
                

        if end_rep - start_rep >= win_len:
            start_rep = start_rep + (end_rep - start_rep - win_len) // 2
            end_rep = start_rep + win_len
            new_rep.append((start_rep, end_rep))
            if siPlot:
                if signal is None:
                    raise ValueError("signal is needed for plotting")
                plt.plot(signal[start_rep:end_rep])
        
        
    return new_act, new_rep

class Network:
    def __init__(self, data, bin, fs=10, ch_names=None, matriz_act=None, matriz_rep=None, densidad_act_prom=None, densidad_rep_prom = None, densidad_act_ch_in=None, densidad_act_ch_out=None, densidad_rep_ch_in=None, densidad_rep_ch_out=None, band = None):
        self.data = data
        self.bin = bin
        self.fs = fs
        self.ch_names = ch_names
        self.band = band
        
    def realizagranger(self, df, maxlag=6, pval=0.01, sel1="HRV", sel2="HRV"):
        try:
            # Verificar si hay suficientes datos
            if len(df) <= maxlag:
                
                print(f"{len(df)} Datos insuficientes?.")
                return 0
            
            # Realizar la prueba de causalidad de Granger
            gc_test_1 = grangercausalitytests(df[[sel1, sel2]], maxlag=maxlag, verbose=False)
            p_values = [gc_test_1[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            
            # Verificar si la media de los p-valores es menor que el umbral
            for val in p_values:
                if val < pval:
                    return 1
            return 0
            #return int(np.mean(p_values) < pval)
            
        except Exception as e:
            print(f"Error en ventana {e}")
            return 0
    
    def crea_matriz(self, df):
        matriz = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i != j:
                    matriz[i, j] = self.realizagranger(df, sel1=df.columns[i], sel2=df.columns[j])
        return matriz
    
    def densidad_red(self, actividades):
        densidades = [np.count_nonzero(actividad) / (actividad.shape[0] * actividad.shape[1]) for actividad in actividades]
        return np.mean(densidades)
    
    def densidad_channel(self, redes, canal, mode="input"):
        densidad_canal = []
        for red in redes:
            if mode == "input":
                densidad_canal.append(np.count_nonzero(red[canal, :]) / red.shape[0])
            elif mode == "output":
                densidad_canal.append(np.count_nonzero(red[:, canal]) / red.shape[1])
        return np.mean(densidad_canal)
    
    def get_ntwks(self, df, bin):
        df = df.fillna(0)
        ventanas_actividad, ventanas_reposo = obtener_win(bin, bin, siPlot=False)
        #ventanas_actividad, ventanas_reposo = selecciona_ventanas(ventanas_actividad, ventanas_reposo)
        for i, ventanas in enumerate(ventanas_reposo):
            if ventanas[1] < ventanas[0]:
                if i == len(ventanas_reposo) - 1:
                    ventanas_reposo = ventanas_reposo[:i]
        
        for i, ventanas in enumerate(ventanas_actividad):
            if ventanas[1] < ventanas[0]:
                if i == len(ventanas_actividad) - 1:
                    ventanas_actividad = ventanas_actividad[:i] 
                                 
        window_len = 15

        diff_actividad = np.diff(ventanas_actividad, axis=1)
        diff_reposo = np.diff(ventanas_reposo, axis=1)

        len_window = window_len * self.fs
        #len_window = int(diff_reposo[diff_reposo > self.fs * window_len].mean())
        
        #max_ventanas = min(len(diff_actividad), len(diff_reposo[diff_reposo > self.fs * window_len]))

        #print(max_ventanas)
        #ventanas_actividad = ventanas_actividad[:max_ventanas]
        #ventanas_reposo = ventanas_reposo[:max_ventanas]

        actividades = []
        for i, ventana in enumerate(ventanas_actividad):
            len_window_temp = ventana[1] - ventana[0]
            if len_window_temp >= len_window:
                fill = (len_window_temp - len_window) // 2
                df_actividad = df.iloc[ventana[0] + fill: ventana[1] - fill, :]
                print("****************************")
                print(ventana[0] + fill, ventana[1] - fill)

                print(df_actividad.shape)
                print("****************************")
                
                matriz = self.crea_matriz(df_actividad)

                actividades.append(matriz)
        actividades = np.array(actividades)
        
        reposos = []
        for i, ventana in enumerate(ventanas_reposo):
            len_window_temp = ventana[1] - ventana[0]
            if len_window_temp >= len_window:
                fill = (len_window_temp - len_window) // 2
                df_reposo = df.iloc[ventana[0] + fill: ventana[1] - fill, :]

                print("****************************")
                print(ventana[0] + fill, ventana[1] - fill)
                
                print(df_reposo.shape)
                print("****************************")
                
                matriz = self.crea_matriz(df_reposo)
                reposos.append(matriz)
                
        reposos = np.array(reposos)
        

        self.matriz_act = np.sum(actividades, axis=0)
        self.matriz_rep = np.sum(reposos, axis = 0)

        self.densidad_act_prom = self.densidad_red(actividades)
        self.densidad_rep_prom = self.densidad_red(reposos)

        self.densidad_act_ch_in = [self.densidad_channel(actividades, i, mode="input") for i in range(actividades[0].shape[0])]
        self.densidad_act_ch_out = [self.densidad_channel(actividades, i, mode="output") for i in range(actividades[0].shape[0])]
        self.densidad_rep_ch_in = [self.densidad_channel(reposos, i, mode="input") for i in range(reposos[0].shape[0])]
        self.densidad_rep_ch_out = [self.densidad_channel(reposos, i, mode="output") for i in range(reposos[0].shape[0])]

        self.array_mat_act = actividades
        self.array_mat_rep = reposos
        return actividades, reposos
    
class Registro:
    def __init__(self, fs):
        """
        Inicializa la clase Registro.

        :param fs: Frecuencia de muestreo.
        """
        self.mu_networks = {}
        self.beta_networks = {}
        self.gamma_networks = {}
        self.fs = fs

    def add_network(self, name, network, tipo):
        """
        Agrega una red a la colección de redes.

        :param name: Nombre de la red.
        :param network: Instancia de la clase Network.
        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        """
        if tipo == 'mu':
            self.mu_networks[name] = network
        elif tipo == 'beta':
            self.beta_networks[name] = network
        elif tipo == 'gamma':
            self.gamma_networks[name] = network
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

    def get_network(self, name, tipo):
        """
        Obtiene una red de la colección de redes.

        :param name: Nombre de la red.
        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        :return: Instancia de la clase Network.
        """
        if tipo == 'mu':
            return self.mu_networks.get(name)
        elif tipo == 'beta':
            return self.beta_networks.get(name)
        elif tipo == 'gamma':
            return self.gamma_networks.get(name)
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

    def remove_network(self, name, tipo):
        """
        Elimina una red de la colección de redes.

        :param name: Nombre de la red.
        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        """
        if tipo == 'mu':
            if name in self.mu_networks:
                del self.mu_networks[name]
        elif tipo == 'beta':
            if name in self.beta_networks:
                del self.beta_networks[name]
        elif tipo == 'gamma':
            if name in self.gamma_networks:
                del self.gamma_networks[name]
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

    def list_networks(self, tipo):
        """
        Lista todas las redes en la colección de redes.

        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        :return: Lista de nombres de las redes.
        """
        if tipo == 'mu':
            return list(self.mu_networks.keys())
        elif tipo == 'beta':
            return list(self.beta_networks.keys())
        elif tipo == 'gamma':
            return list(self.gamma_networks.keys())
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

from collections import deque


def bfs_shortest_paths(graph, start):
    """Breadth-First Search to find all shortest paths from start node in a directed graph."""
    num_nodes = graph.shape[0]
    dist = [-1] * num_nodes
    dist[start] = 0
    paths = [[] for _ in range(num_nodes)]
    paths[start] = [[start]]
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        for neighbor in range(num_nodes):
            if graph[current, neighbor] > 0:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
                if dist[neighbor] == dist[current] + 1:
                    for path in paths[current]:
                        paths[neighbor].append(path + [neighbor])
    
    return paths

def betweenness_centrality(graph):
    num_nodes = graph.shape[0]
    betweenness = np.zeros(num_nodes)
    
    for s in range(num_nodes):
        paths = bfs_shortest_paths(graph, s)
        for t in range(num_nodes):
            if s != t:
                num_paths = len(paths[t])
                if num_paths > 0:
                    node_counts = np.zeros(num_nodes)
                    for path in paths[t]:
                        for node in path[1:-1]:
                            node_counts[node] += 1
                    betweenness += node_counts / num_paths
    
    return betweenness

def get_deg(test_mat):
    temp_in_deg = np.zeros((test_mat.shape[0]))
    temp_out_deg = np.zeros((test_mat.shape[0]))
    
    for i in range(test_mat.shape[0]):
        temp_in_deg[i] = np.sum(test_mat[:, i]) / (test_mat.shape[0] - 1)
        temp_out_deg[i] = np.sum(test_mat[i, :]) / (test_mat.shape[0] - 1)
    return temp_in_deg, temp_out_deg