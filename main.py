import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np
from motor_imunologico import treinar_clonalg, calcular_afinidade

class ClassificadorCLONALG:
    def __init__(self):
        # Vai guardar o anticorpo perfeito para Setosa(0), Versicolor(1) e Virginica(2)
        self.memoria_imunologica = {} 

    def fit(self, X_train, y_train):
        """Treina um exército para cada espécie de flor."""
        classes = np.unique(y_train)
        
        for cls in classes:
            print(f"Treinando o Sistema Imunológico para a espécie: {cls}...")
            
            # Filtra apenas as flores dessa classe específica (transformando em numpy array)
            flores_da_classe = X_train[y_train == cls].values
            
            # Chama o SEU motor para achar o anticorpo perfeito para essa flor
            melhor_anticorpo = treinar_clonalg(flores_da_classe, num_geracoes=50, tam_populacao=50, num_clones = 5)
            
            # Salva o anticorpo na memória
            self.memoria_imunologica[cls] = melhor_anticorpo
            
        print("Treinamento Imunológico Concluído!\n")

    def predict(self, X_test):
        """O algoritmo vê flores novas e tenta adivinhar quais são baseadas na memória."""
        y_pred = []
        
        # Para cada flor no conjunto de teste...
        for _, flor in X_test.iterrows():
            medidas_flor = flor.values
            afinidades = {}
            
            # ...ele testa a afinidade contra as 3 memórias imunológicas
            for cls, anticorpo_memoria in self.memoria_imunologica.items():
                afinidade = calcular_afinidade(medidas_flor, anticorpo_memoria)
                afinidades[cls] = afinidade
                
            # A memória que tiver a MAIOR afinidade vence (é a espécie prevista)
            classe_vencedora = max(afinidades, key=afinidades.get)
            y_pred.append(classe_vencedora)
            
        return y_pred

def carregar_dados():
    """Carrega o dataset e retorna o DataFrame e os dados originais do Iris."""
    iris_data = load_iris(as_frame=True)

    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['target'] = iris_data.target
    
    return df, iris_data

def preparar_dados(df):
    """Separa as características e o alvo e divide em dados de treino e teste."""
    # Define as características (X) e o alvo (y)
    # n_features = 4 e n_classes = 3 
    X = df.drop('target', axis=1) 
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    
    return X_train, X_test, y_train, y_test

def avaliar_modelo(y_test, y_pred, target_names):
    """Calcula as métricas e imprime o relatório de classificação do modelo."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro') # 'macro' para problemas de 3 classes
    rec = recall_score(y_test, y_pred, average='macro')

    print(f"Acurácia Final: {acc:.2%}")
    print(f"Precisão: {prec:.2%}")
    print(f"Recall: {rec:.2%}")

 
    print("\nRelatório de Classificação Completo:")
    print(classification_report(y_test, y_pred, target_names=target_names))

def main():
    df, iris_data = carregar_dados()

    print("Visualização das primeiras linhas do dataset:")
    print(df.head())

    X_train, X_test, y_train, y_test = preparar_dados(df)

    # 1. Inicializa a nossa classe "Cola"
    seu_modelo_clonalg = ClassificadorCLONALG()
    
    # 2. Inicia o Treinamento (Onde a mágica do seu motor acontece)
    seu_modelo_clonalg.fit(X_train, y_train)
    
    # 3. Faz as previsões no conjunto de teste invisível
    y_pred = seu_modelo_clonalg.predict(X_test)

    # 4. Avalia e imprime as métricas que o professor pediu!
    avaliar_modelo(y_test, y_pred, iris_data.target_names)

if __name__ == "__main__":
    main()
