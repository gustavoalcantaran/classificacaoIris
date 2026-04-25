import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    
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

    # Inicializar e treinar o modelo aqui
    # seu_modelo_clonalg = ...
    
    # Suponha que 'y_test' são as respostas reais (as flores verdadeiras)
    # E 'y_pred' são as respostas que o seu algoritmo CLONALG gerou para o teste
    y_pred = seu_modelo_clonalg.predict(X_test)

    # Quando tiver a variável y_pred (após a predição), descomente a linha abaixo para ver as métricas:
    # avaliar_modelo(y_test, y_pred, iris_data.target_names)

if __name__ == "__main__":
    main()
