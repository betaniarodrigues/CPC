from time import time
import csv
import matplotlib.pyplot as plt
from arguments import parse_args
from utils import set_all_seeds
import pandas as pd
from evaluate_with_classifier import evaluate_with_classifier 

plt.rcParams['font.family'] = 'serif'

def plot_f1_curves(results_csv, output_path='f1_scores_plot.png'):
    """
    Plota as curvas de F1-score para cada dataset e modelo (com e sem pesos pré-treinados).
    :param results_csv: Caminho para o arquivo CSV com os resultados
    :param output_path: Caminho para salvar o arquivo PNG com os gráficos
    """
    # Carregar os resultados do CSV
    df = pd.read_csv(results_csv)

    # # Definir as porcentagens de dados
    # percentages = [10, 50, 100]

    # Obter os datasets disponíveis no arquivo CSV
    datasets = df['dataset'].unique()

    # Criar uma figura com subplots (um gráfico por dataset)
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(8, 6 * num_datasets))

    if num_datasets == 1:
        axes = [axes]  # Garantir que seja uma lista mesmo com um único dataset

    # Iterar sobre cada dataset e gerar as curvas
    for i, dataset in enumerate(datasets):
        ax = axes[i]

        # Filtrar os dados do dataset atual
        df_dataset = df[df['dataset'] == dataset]

        # Separar os resultados com e sem pesos pré-treinados
        df_pretrained = df_dataset[df_dataset['pretrained'] == 'Yes']
        df_no_pretrained = df_dataset[df_dataset['pretrained'] == 'No']

        # Plotar a curva para "Sem CPC" (sem pesos pré-treinados)
        ax.plot(df_no_pretrained['data_percentage'], df_no_pretrained['test_f1_score'], 
                marker='x', linestyle='--', color='blue', label='Sem CPC')

        # Plotar a curva para "Com CPC" (com pesos pré-treinados)
        ax.plot(df_pretrained['data_percentage'], df_pretrained['test_f1_score'], 
                marker='o', linestyle='-', color='green', label='Com CPC')

        # Adicionar título e legendas
        ax.set_title(f'F1-Score vs Percentage of Data Used for {dataset}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Percentage of Data Used (%)', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.grid(True)
        ax.legend()

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()

    # Salvar a figura como um arquivo PNG
    plt.savefig(output_path)

    print(f'Plot saved as {output_path}')

def run_experiments(args):
    """
    Run experiments for multiple datasets, data percentages, with and without pretrained weights.
    :param args: Arguments
    :return: None
    """
    datasets = ["UCI_raw", "KuHar_raw", "MotionSense_raw", "RealWorld_raw"]  # Lista de datasets a serem testados
    data_percentages = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Variações de porcentagem de dados 

    results_csv = 'experiments_results.csv'
    
    # Abrindo um arquivo CSV para salvar os resultados
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escrevendo o cabeçalho
        writer.writerow(['dataset', 'data_percentage', 'pretrained', 'val_loss', 'test_accuracy', 'test_f1_score'])

        for dataset in datasets:
            for percentage in data_percentages:
                print(f"\nRunning experiment for dataset: {dataset}, with {percentage}% of data.")

                # Atualizando os argumentos para o dataset e porcentagem atuais
                args.dataset = dataset
                
                if dataset == "UCI_raw":
                    args.data_file = "UCI_raw"
                    args.num_classes = 7
                elif dataset == "KuHar_raw":
                    args.data_file = "KuHar_raw"
                    args.num_classes = 18
                elif dataset == "MotionSense_raw":
                    args.data_file = "MotionSense_raw"
                    args.num_classes = 6
                elif dataset == "RealWorld_raw":
                    args.data_file = "RealWorld_raw"
                    args.num_classes = 9
                
                args.data_percentage = percentage

                # Configurando as seeds
                set_all_seeds(args)

                # 1. Rodar sem pesos pré-treinados (saved_model=None)
                if args.saved_model == None:
                    print(f"Running experiment WITHOUT pretrained weights for {dataset} with {percentage}% data.")
                    results_no_pretrained = evaluate_with_classifier(args)

                    # Escrevendo os resultados no CSV para o experimento sem pesos pré-treinados
                    writer.writerow([dataset, percentage, 'No', 
                                    results_no_pretrained['val_loss'], 
                                    results_no_pretrained['test_accuracy'], 
                                    results_no_pretrained['test_f1_score']])

                    print(f"Experiment WITHOUT pretrained weights for {dataset} with {percentage}% data completed.")

                # 2. Rodar com pesos pré-treinados, se fornecido
                else:
                    # Rodar com pesos pré-treinados
                    print(f"Running experiment WITH pretrained weights for {dataset} with {percentage}% data.")
                    results_pretrained = evaluate_with_classifier(args)

                    # Escrevendo os resultados no CSV para o experimento com pesos pré-treinados
                    writer.writerow([dataset, percentage, 'Yes', 
                                    results_pretrained['val_loss'], 
                                    results_pretrained['test_accuracy'], 
                                    results_pretrained['test_f1_score']])

                    print(f"Experiment WITH pretrained weights for {dataset} with {percentage}% data completed.")

                    # Guardar o valor original de args.saved_model
                    saved_model_original = args.saved_model

                    # Rodar sem pesos pré-treinados
                    print(f"Running experiment WITHOUT pretrained weights for {dataset} with {percentage}% data (even though pretrained was provided).")
                    args.saved_model = None  # Setar para None para rodar sem pesos
                    results_no_pretrained = evaluate_with_classifier(args)

                    # Escrevendo os resultados no CSV para o experimento sem pesos pré-treinados
                    writer.writerow([dataset, percentage, 'No', 
                                    results_no_pretrained['val_loss'], 
                                    results_no_pretrained['test_accuracy'], 
                                    results_no_pretrained['test_f1_score']])

                    print(f"Experiment WITHOUT pretrained weights for {dataset} with {percentage}% data completed.")

                    # Restaurar o valor original de args.saved_model para as próximas iterações
                    args.saved_model = saved_model_original
    
    plot_f1_curves(results_csv)


if __name__ == '__main__':
    # Parsing arguments
    args = parse_args()

    # Executando os experimentos
    run_experiments(args)

    print('------ All experiments complete! ------')
