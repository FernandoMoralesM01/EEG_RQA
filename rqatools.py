from scipy.spatial import distance
import numpy as np
from scipy.stats import multivariate_normal

import logging
logging.basicConfig(level=logging.CRITICAL)


def get_miniRR(row, means, vecindad, alpha=10, recurrence_rate=None, distance_vecindad=None, order=2):
    try:
        if recurrence_rate is None:
            recurrence_rate = [0]
        if distance_vecindad is None:
            distance_vecindad = []

        # Calcular distancia euclidiana al centroide promedio
        dist = distance.euclidean(row, means)

        # Asegurarse de que `sigmas` es escalar (std), si es una matriz usar alguna métrica
        if isinstance(vecindad, np.ndarray) and vecindad.ndim == 2:
            vecindad = np.sqrt(np.trace(vecindad)) * alpha
        else:
            vecindad = vecindad * alpha

        # Evaluar si está dentro de la vecindad
        in_neighborhood = 1 if dist < vecindad else 0
        distance_vecindad.append(in_neighborhood)

        # Evaluar recurrencia
        for i in range(order):
            if len(distance_vecindad) > i and distance_vecindad[-(i + 1)] == 1 and distance_vecindad[-i] == 1:
                recurrence_rate.append(recurrence_rate[-1] + 1)
                break
            elif i == order - 1:
                recurrence_rate.append(0)

        return recurrence_rate, distance_vecindad
    except:
        return recurrence_rate, distance_vecindad



def EM_estimation(row, n_componentes = 2, mus = None, sigmas = None, pis = None, alpha = 0.01):
    '''
        row: (rasgos, 0)
        n_componentes: número de clusters
        alpha: learining rate

        initial_mus: medias iniciales del sistema
        initial_sigmas: desviaciones iniciales del sistema
        initial_pis: pis iniciales del sistema

        Estos valores se pueden iniciar en rando o se puede usar una ventana chica para estimar los centros más rápidamente 

            initial_data = data[:, :10].T 
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(initial_data)
            mus = gmm.means_
            sigmas = gmm.covariances_
            pis = gmm.weights_

    '''
    mus = np.copy(mus)
    sigmas = np.copy(sigmas)
    pis = np.copy(pis)

    gamma = np.array([
        pis[k] * multivariate_normal.pdf(row, mean=mus[k], cov=sigmas[k])
        for k in range(n_componentes)
    ])
    
    gamma /= gamma.sum()

    # Step 3: M-step

    for k in range(n_componentes):
        diff = row - mus[k]
        mus[k] += alpha * gamma[k] * diff
        sigmas[k] += alpha * gamma[k] * (np.outer(diff, diff) - sigmas[k])
        pis[k] += alpha * (gamma[k] - pis[k])
        
    
    return mus, sigmas, pis, gamma



def K_means_estimation(row, k = 2, centroids = [], alpha = 0.01):
    '''
        k: Numero de clusters
        centroids: posicion inicial de los centroides
        alpha: learning rate
    '''

    dists = [np.linalg.norm(row - c) for c in centroids]
    
    # Paso 2: asignar al más cercano
    closest = np.argmin(dists)
    
    # Paso 3: actualizar ese centroide
    centroids[closest] += alpha * (row - centroids[closest])
    
    return centroids


def neighborhood(bmu_index, n_neurons, sigma):
    d = np.abs(np.arange(n_neurons) - bmu_index)
    return np.exp(-d**2 / (2 * sigma**2))

def SOM_estimation(row, n_neurons = 2, alpha = 0.01, sigma = 0.5, weights = None):
    
    # 1. Buscar la Best Matching Unit (BMU)
    dists = np.linalg.norm(weights - row, axis=1)
    bmu_idx = np.argmin(dists)

    # 2. Actualizar los pesos de todas las neuronas
    h = neighborhood(bmu_idx, n_neurons, sigma)
    for i in range(n_neurons):
        weights[i] += alpha * h[i] * (row - weights[i])
    
    return weights
    