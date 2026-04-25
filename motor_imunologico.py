import numpy as np
import random

def inicializar_anticorpos(tamanho_populacao, num_medidas=4):
    """
    Gera anticorpos aleatórios. 
    As flores Iris têm medidas variando de ~0.1 a ~8.0.
    """
    return np.random.uniform(0.1, 8.0, size=(tamanho_populacao, num_medidas))

def calcular_afinidade(antigeno, anticorpo):
    """
    Calcula a afinidade usando a Distância Euclidiana.
    Quanto menor a distância (mais parecidos), maior a afinidade (perto de 1.0).
    """
    distancia = np.linalg.norm(np.array(antigeno) - np.array(anticorpo))
    return 1.0 / (1.0 + distancia)

def calcular_afinidade_global(antigenos, anticorpo):
    """
    CORREÇÃO AQUI: Calcula a afinidade média do anticorpo contra TODAS 
    as flores da classe. Isso força ele a ir para o centro do grupo!
    """
    afinidades = [calcular_afinidade(ag, anticorpo) for ag in antigenos]
    return sum(afinidades) / len(afinidades)

def hipermutacao(anticorpo, afinidade, taxa_base=1.0):
    """
    A Mágica do CLONALG: Mutação Inversamente Proporcional.
    Se a afinidade for alta (0.9), a mutação é minúscula (0.1).
    Se a afinidade for baixa (0.1), a mutação é gigante (0.9).
    """
    forca_mutacao = (1.0 - afinidade) * taxa_base
    
    anticorpo_mutado = anticorpo.copy()
    
    for i in range(len(anticorpo_mutado)):
        # Aplica o ruído Gaussiano em cada medida da flor
        anticorpo_mutado[i] += random.gauss(0, forca_mutacao)
        # Trava para impedir que a flor tenha uma medida negativa
        anticorpo_mutado[i] = max(0.1, anticorpo_mutado[i]) 
        
    return anticorpo_mutado

def treinar_clonalg(antigenos_treino, num_geracoes, tam_populacao, num_clones):
    """Loop Principal do Motor Imunológico."""
    anticorpos = inicializar_anticorpos(tam_populacao, num_medidas=len(antigenos_treino[0]))
    
    for geracao in range(num_geracoes):
        nova_populacao = []
        
        # Expoe os anticorpos a todos os vírus para gerar clones
        for antigeno in antigenos_treino:
            afinidades = [calcular_afinidade(antigeno, ant) for ant in anticorpos]
            indices_melhores = np.argsort(afinidades)[-3:] 
            
            for idx in indices_melhores:
                ant_campeao = anticorpos[idx]
                af_campeao = afinidades[idx]
                nova_populacao.append(ant_campeao)
                
                for _ in range(num_clones):
                    clone_mutado = hipermutacao(ant_campeao, af_campeao)
                    nova_populacao.append(clone_mutado)
                    
        afinidades_finais = [calcular_afinidade_global(antigenos_treino, ant) for ant in nova_populacao]
        indices_sobreviventes = np.argsort(afinidades_finais)[-tam_populacao:]
        
        anticorpos = np.array([nova_populacao[i] for i in indices_sobreviventes])
        
    # Retorna o anticorpo que ficou mais centralizado na nuvem daquela espécie
    afinidades_finais = [calcular_afinidade_global(antigenos_treino, ant) for ant in anticorpos]
    melhor_indice_absoluto = np.argmax(afinidades_finais)
    
    return anticorpos[melhor_indice_absoluto]