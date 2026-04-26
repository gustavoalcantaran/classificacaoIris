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
    Quanto menor a distância (mais parecidos), maior a afinidade.
    """
    distancia = np.linalg.norm(np.array(antigeno) - np.array(anticorpo))
    # distancia = sqrt((a1 - b1)^2) + (a2 - b2)^2 + ... + (a4 - b4)^2)
    return 1.0 / (1.0 + distancia)

def calcular_afinidade_global(antigenos, anticorpo):
    """
    Calcula a afinidade média do anticorpo contra todas as flores da classe.
    Isso serve como função de aptidão (fitness) para o algoritmo.
    """
    if len(antigenos) == 0:
        return 0
    afinidades = [calcular_afinidade(ag, anticorpo) for ag in antigenos]
    return sum(afinidades) / len(afinidades)

def hipermutacao(anticorpo, afinidade, afinidade_max=1.0, taxa_base=1.0):
    """
    Mutação Inversamente Proporcional à Afinidade.
    Fórmula exata: hiperAntik = (1 - (afinik/afinmax)) * beta
    """
    if afinidade_max <= 0:
        afinidade_max = 1.0
        
    # beta = taxa_base
    forca_mutacao = (1.0 - (afinidade / afinidade_max)) * taxa_base
    
    # Se a afinidade for igual à máxima, forca_mutacao será 0. Garantimos uma pequena mutação para diversidade.
    forca_mutacao = max(0.01, forca_mutacao)
    
    anticorpo_mutado = anticorpo.copy()
    
    for i in range(len(anticorpo_mutado)):
        # Aplica o ruído Gaussiano
        anticorpo_mutado[i] += random.gauss(0, forca_mutacao)
        # Trava para impedir que a flor tenha uma medida muito baixa
        anticorpo_mutado[i] = max(0.1, anticorpo_mutado[i]) 
        
    return anticorpo_mutado

def evoluir_geracao(anticorpos, antigenos_treino, tam_populacao, num_clones=None, n2_novos=5):
    """
    Executa uma geração do CLONALG.
    Passos: Avaliação -> Seleção -> Clonagem -> Hipermutação -> Metadinâmica.
    Retorna a nova população de tamanho `tam_populacao`.
    """
    # 1. Avaliação de aptidão
    afinidades = [calcular_afinidade_global(antigenos_treino, ant) for ant in anticorpos]
    afinidade_max = 1.0 # Máximo teórico para distância = 0
    
    # 2. Seleção (M elementos de maior aptidão) - Ex: metade superior
    M = max(1, tam_populacao // 2)
    indices_melhores = np.argsort(afinidades)[-M:]
    
    afinidades_selecionados = [afinidades[idx] for idx in indices_melhores]
    soma_afinidades = sum(afinidades_selecionados)
    
    nova_populacao = []
    
    # 3 e 4. Clonagem e Maturação por afinidade (Hipermutação)
    # A população será P-N clones e N novos. Então Total de Clones Cl = tam_populacao - n2_novos.
    total_clones = tam_populacao - n2_novos
    clones_gerados = 0
    
    for i, idx in enumerate(indices_melhores):
        ant_campeao = anticorpos[idx]
        af_campeao = afinidades[idx]
        
        # Fórmula exata: QC = (af_k / sum(af)) * Cl
        if soma_afinidades > 0:
            qc = int(round((af_campeao / soma_afinidades) * total_clones))
        else:
            qc = total_clones // M
            
        # O último da lista recebe o saldo exato para garantir o número total de clones
        if i == len(indices_melhores) - 1:
            qc = max(0, total_clones - clones_gerados)
            
        clones_gerados += qc
        
        for _ in range(qc):
            clone_mutado = hipermutacao(ant_campeao, af_campeao, afinidade_max=afinidade_max, taxa_base=1.0)
            nova_populacao.append(clone_mutado)
            
    # Garantir estritamente o tamanho total_clones (devido a eventuais erros de arredondamento atípicos)
    if len(nova_populacao) > total_clones:
        nova_populacao = nova_populacao[:total_clones]
    while len(nova_populacao) < total_clones:
        ant_campeao = anticorpos[indices_melhores[-1]]
        af_campeao = afinidades[indices_melhores[-1]]
        clone_mutado = hipermutacao(ant_campeao, af_campeao, afinidade_max=afinidade_max, taxa_base=1.0)
        nova_populacao.append(clone_mutado)
    
    # 5. Metadinâmica: adicionar N novos gerados aleatoriamente
    if n2_novos > 0:
        novos = inicializar_anticorpos(n2_novos, num_medidas=len(antigenos_treino[0]))
        proxima_geracao = nova_populacao + list(novos)
    else:
        proxima_geracao = nova_populacao
        
    return np.array(proxima_geracao)

def extrair_melhor_anticorpo(anticorpos, antigenos_treino):
    """
    Identifica qual anticorpo da população melhor representa a classe.
    """
    afinidades_finais = [calcular_afinidade_global(antigenos_treino, ant) for ant in anticorpos]
    melhor_indice_absoluto = np.argmax(afinidades_finais)
    return anticorpos[melhor_indice_absoluto]