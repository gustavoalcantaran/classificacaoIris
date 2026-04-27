import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Importa as funções atualizadas do motor imunológico
from motor_imunologico import inicializar_anticorpos, evoluir_geracao, calcular_afinidade, extrair_melhor_anticorpo

class ClassificadorCLONALG:
    """
    Orquestra as populações de anticorpos para cada classe do problema.
    Mantém o estado da evolução.
    """
    def __init__(self, tam_populacao=50, num_clones=5):
        self.tam_populacao = tam_populacao
        self.num_clones = num_clones
        self.populacoes = {} 
        self.memoria_imunologica = {} 
        self.classes = []

    def inicializar(self, X_train, y_train):
        """Cria as populações iniciais para cada classe."""
        self.classes = np.unique(y_train)
        for cls in self.classes:
            flores_da_classe = X_train[y_train == cls]
            # Usa shape[1] para inferir o número de características (medidas)
            self.populacoes[cls] = inicializar_anticorpos(self.tam_populacao, num_medidas=X_train.shape[1])
            self.memoria_imunologica[cls] = extrair_melhor_anticorpo(self.populacoes[cls], flores_da_classe)

    def treinar_uma_geracao(self, X_train, y_train):
        """Evolui a população de cada classe por uma única geração."""
        for cls in self.classes:
            flores_da_classe = X_train[y_train == cls]
            # Metadinâmica: 10% da população é renovada com novos anticorpos
            n2_novos = max(1, int(0.1 * self.tam_populacao)) 
            
            nova_pop = evoluir_geracao(
                anticorpos=self.populacoes[cls], 
                antigenos_treino=flores_da_classe, 
                tam_populacao=self.tam_populacao, 
                num_clones=self.num_clones, 
                n2_novos=n2_novos
            )
            self.populacoes[cls] = nova_pop
            
            # Atualiza o melhor anticorpo de memória para uso no predict
            self.memoria_imunologica[cls] = extrair_melhor_anticorpo(nova_pop, flores_da_classe)

    def predict(self, X_test):
        """Prevê a classe baseada na maior afinidade com a memória imunológica."""
        y_pred = []
        for flor in X_test:
            afinidades = {}
            for cls, anticorpo_memoria in self.memoria_imunologica.items():
                afinidade = calcular_afinidade(flor, anticorpo_memoria)
                afinidades[cls] = afinidade
            # A classe prevista é a que tem o anticorpo de memória mais afim
            classe_vencedora = max(afinidades, key=afinidades.get)
            y_pred.append(classe_vencedora)
        return y_pred


def experimento_acuracia_por_geracao(X_train, X_test, y_train, y_test, num_geracoes=50, tam_populacao=40):
    """
    Experimento 1: Treina o modelo passo a passo e exibe como a acurácia de teste
    varia no decorrer das gerações.
    """
    print("--------------------------------------------------")
    print(f"Iniciando Experimento 1: Acurácia ao longo de {num_geracoes} gerações...")
    print(f"Tamanho da População: {tam_populacao}")
    print("--------------------------------------------------")
    
    modelo = ClassificadorCLONALG(tam_populacao=tam_populacao, num_clones=5)
    modelo.inicializar(X_train, y_train)
    
    historico_acc = []
    
    for g in range(num_geracoes):
        modelo.treinar_uma_geracao(X_train, y_train)
        
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        historico_acc.append(acc)
        # print(f"Geração {g+1}/{num_geracoes} - Acurácia: {acc:.2%}")
        
    y_pred_train = modelo.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print(f"Final do treinamento - Acurácia no Treino: {acc_train:.2%}")
    print(f"Final do treinamento - Acurácia no Teste: {historico_acc[-1]:.2%}")
    
    # Gerar Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_geracoes + 1), historico_acc, marker='o', linestyle='-', color='b', markersize=4)
    plt.title("Evolução da Acurácia no Conjunto de Teste")
    plt.xlabel("Geração")
    plt.ylabel("Acurácia (Teste)")
    plt.grid(True)
    plt.savefig('grafico_acuracia_geracao.png')
    plt.close()
    print("-> Gráfico gerado com sucesso: 'grafico_acuracia_geracao.png'\n")


def experimento_tamanho_populacao(X_train, X_test, y_train, y_test, num_geracoes=50):
    """
    Experimento 2: Analisa o impacto do tamanho da população na acurácia final.
    Varia a população entre 10 e 50 (passo 10) e utiliza 50 gerações.
    """
    print("--------------------------------------------------")
    print("Iniciando Experimento 2: Impacto do Tamanho da População...")
    print("--------------------------------------------------")
    
    tamanhos = [10, 20, 30, 40, 50]
    historico_acc_final = []
    
    for tam in tamanhos:
        # print(f"Testando população de tamanho {tam}...")
        modelo = ClassificadorCLONALG(tam_populacao=tam, num_clones=5)
        modelo.inicializar(X_train, y_train)
        
        for _ in range(num_geracoes):
            modelo.treinar_uma_geracao(X_train, y_train)
            
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        historico_acc_final.append(acc)
        print(f"População = {tam:2d} -> Acurácia Teste: {acc:.2%}")
        
    # Gerar Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(tamanhos, historico_acc_final, marker='s', linestyle='--', color='darkorange', markersize=8)
    plt.title(f"Impacto do Tamanho da População na Acurácia (após {num_geracoes} gerações)")
    plt.xlabel("Tamanho da População")
    plt.ylabel("Acurácia Final (Teste)")
    plt.grid(True)
    plt.xticks(tamanhos)
    
    # Define o eixo Y com alguma margem para facilitar visualização
    plt.ylim([max(0, min(historico_acc_final) - 0.05), 1.05])
    
    plt.savefig('grafico_impacto_populacao.png')
    plt.close()
    print("\n-> Gráfico gerado com sucesso: 'grafico_impacto_populacao.png'\n")


def carregar_dados():
    """Carrega o dataset e retorna os dados brutos e os alvos."""
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    return X, y

def main():
    X, y = carregar_dados()

    # Divisão treino e teste (30% para teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y
    )

    # Executa os dois experimentos solicitados pelo professor
    experimento_acuracia_por_geracao(X_train, X_test, y_train, y_test, num_geracoes=50, tam_populacao=50)
    experimento_tamanho_populacao(X_train, X_test, y_train, y_test, num_geracoes=50)

if __name__ == "__main__":
    main()
