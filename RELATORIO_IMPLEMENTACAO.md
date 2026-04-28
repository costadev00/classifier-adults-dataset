# Relatório de Implementação - Adult Income

Este relatório resume o fluxo implementado no projeto, desde a preparação dos dados até a comparação final entre **kNN**, **SVM linear** e **Regressão Logística**. O foco aqui não é detalhar cada função ou cada parâmetro de biblioteca, mas explicar as decisões que fizeram diferença no resultado e na confiabilidade da comparação.

O ponto central do trabalho foi construir uma comparação justa entre os classificadores: todos foram avaliados com os mesmos dados preparados, o mesmo conjunto de variáveis, a mesma divisão entre desenvolvimento e holdout, o mesmo protocolo de validação cruzada e a mesma métrica de seleção.

## Linha do tempo pelos commits

O histórico de commits conta bem a evolução do projeto:

| Commit | Etapa | Papel na implementação |
|---|---|---|
| `4a84aed` | Download e início da preparação | Entrada dos arquivos originais da competição, `train_data.csv`, `test_data.csv`, `sample_submission.csv` e documentação auxiliar. |
| `38ef825` | Diagnóstico dos dados | Criação do notebook de qualidade dos dados, usado para orientar decisões conservadoras de limpeza e engenharia. |
| `71f23f0` | Preparação dos dados | Criação de `train_prepared.csv`, `test_prepared.csv`, arquivos auxiliares e manifesto de preparação. |
| `8a558d9` | Primeiro kNN | Primeiro classificador completo com artefatos de métricas, parâmetros e submissão. |
| `a482c88` | kNN refinado | Melhoria metodológica e explicativa do notebook kNN, incluindo avaliação mais completa. |
| `d9e7517` | SVM implementado | Segundo classificador, comparável ao kNN pelo mesmo protocolo. |
| `9f25417` | Regressão Logística finalizada | Terceiro classificador, comparação consolidada dos três modelos e gráfico final. |
| `0866a4d` | Documentação da Regressão Logística | Nota conceitual conectando teoria da disciplina com implementação prática no scikit-learn. |

Essa sequência mostra uma evolução gradual: primeiro entender e preparar a base, depois construir um baseline, em seguida melhorar a metodologia, e por fim comparar modelos de forma padronizada.

## 1. Preparação dos Dados

A preparação foi consolidada no notebook `notebooks/02_data_preparation.ipynb`. Ela transformou os arquivos originais da competição em uma interface mais segura e reutilizável para modelagem:

- `data/prepared/adult_income/train_prepared.csv`
- `data/prepared/adult_income/test_prepared.csv`
- `data/prepared/adult_income/sample_submission.csv`
- artefatos auxiliares descritos em `data/prepared/adult_income/manifest.json`

A decisão mais importante nessa etapa foi **não tratar a preparação como uma transformação única e opaca**. O notebook manteve um fluxo explícito: diagnóstico, limpeza conservadora, engenharia de variáveis e exportação de visões adequadas para modelos diferentes.

Alguns pontos que fizeram diferença:

- Remoção apenas de duplicatas exatas com mesmo alvo, evitando descarte agressivo de dados.
- Preservação de casos conflitantes para diagnóstico, em vez de apagá-los silenciosamente.
- Criação de variáveis transformadas com `log1p` para colunas muito assimétricas, como `fnlwgt`, `capital.gain` e `capital.loss`.
- Criação de indicadores de ausência para variáveis categóricas com valores faltantes.
- Separação entre uma visão mais adequada para árvores e uma visão mais adequada para modelos lineares ou baseados em distância.
- Manutenção de arquivos CSV finais com uma interface simples para os notebooks seguintes.

Em números, o treino original tinha 32.560 registros de dados mais cabeçalho no CSV, enquanto `train_prepared.csv` ficou com 32.536 registros de dados mais cabeçalho. A diferença veio da remoção conservadora de duplicatas exatas. A base de teste preparada manteve 16.280 registros de dados mais cabeçalho.

O resultado mais importante da preparação foi permitir que os modelos seguintes fossem treinados sem repetir decisões de limpeza, reduzindo risco de inconsistência entre notebooks.

## Célula que mais fez diferença no fluxo

A célula abaixo foi a que considero mais importante para a comparação final. Ela aparece nos notebooks de modelagem e fixa o mesmo conjunto compacto de features para os três classificadores. Essa escolha evitou que um modelo fosse favorecido por receber variáveis diferentes dos outros.

Ela também deixa explícito o que ficou fora da modelagem: identificador, alvo, variáveis brutas substituídas por versões transformadas e variáveis derivadas que poderiam ampliar demais o espaço de atributos.

```python
NUMERIC_FEATURES = [
    "age",
    "education.num",
    "hours.per.week",
    "log1p_fnlwgt",
    "log1p_capital_gain",
    "log1p_capital_loss",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

SELECTED_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_FEATURE_COLUMNS = [
    ID_COL,
    TARGET_COL,
    "fnlwgt",
    "capital.gain",
    "capital.loss",
    "age_band",
    "hours_ge_50",
    "hours_ge_60",
    "hours_ge_80",
    "capital_gain_positive",
    "capital_loss_positive",
    "workclass_missing",
    "occupation_missing",
    "native_country_missing",
    "education_level_group",
    "family_role_combo",
]

missing_train_features = sorted(set(SELECTED_FEATURE_COLUMNS) - set(train_prepared.columns))
missing_test_features = sorted(set(SELECTED_FEATURE_COLUMNS) - set(test_prepared.columns))
assert not missing_train_features, f"Missing selected features in train_prepared.csv: {missing_train_features}"
assert not missing_test_features, f"Missing selected features in test_prepared.csv: {missing_test_features}"

X_train_full = train_prepared[SELECTED_FEATURE_COLUMNS].copy()
X_test_final = test_prepared[SELECTED_FEATURE_COLUMNS].copy()
y_binary = (train_prepared[TARGET_COL] == POSITIVE_LABEL).astype(int)

feature_summary_df = pd.DataFrame(
    {
        "feature": SELECTED_FEATURE_COLUMNS,
        "feature_type": ["numeric"] * len(NUMERIC_FEATURES) + ["categorical"] * len(CATEGORICAL_FEATURES),
    }
)

excluded_feature_summary_df = pd.DataFrame({"excluded_feature": EXCLUDED_FEATURE_COLUMNS})

target_distribution_df = pd.DataFrame(
    {
        "income": train_prepared[TARGET_COL].value_counts().index,
        "count": train_prepared[TARGET_COL].value_counts().values,
        "share": train_prepared[TARGET_COL].value_counts(normalize=True).round(4).values,
    }
)

display(feature_summary_df)
display(excluded_feature_summary_df)
display(target_distribution_df)
```

Essa célula fez diferença porque amarrou a comparação em uma base comum. Sem ela, seria fácil comparar modelos que, na prática, estariam vendo problemas diferentes.

## 2. kNN

O kNN foi o primeiro classificador completo do projeto. Ele serviu como baseline metodológico e também como ponto de partida para estruturar os artefatos dos outros modelos.

O notebook `notebooks/03_knn_classifier.ipynb` estabeleceu o padrão principal:

- separar treino rotulado em development e holdout;
- usar validação cruzada apenas no development;
- deixar o holdout completamente fora da escolha de hiperparâmetros;
- salvar parâmetros escolhidos, métricas de holdout e submissão;
- explicar visualmente a lógica do modelo.

O kNN é muito sensível à escala e ao excesso de dimensões. Por isso, a decisão de usar um conjunto compacto de features foi especialmente importante aqui. As variáveis numéricas foram escaladas e as categóricas foram codificadas, permitindo que a noção de distância fosse minimamente coerente.

O melhor kNN selecionado foi:

- `k = 61`
- `weights = uniform`
- `p = 1`, distância Manhattan
- média de acurácia em validação cruzada: `0.84912`

No holdout, o kNN obteve:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.84819 |
| Balanced accuracy | 0.77354 |
| Precision positiva | 0.70803 |
| Recall/Sensitivity positiva | 0.62946 |
| Specificity | 0.91761 |
| F1 positivo | 0.66644 |
| ROC AUC | 0.90215 |

O resultado mais relevante do kNN não foi apenas a acurácia. Ele foi o modelo com melhor **balanced accuracy**, melhor **recall/sensitivity**, melhor **F1 positivo** e melhor **ROC AUC** entre os três. Isso indica que ele lidou melhor com a classe positiva `>50K`, que é a classe minoritária e mais difícil.

## 3. SVM Linear

Depois do kNN, o notebook `notebooks/04_svm_classifier.ipynb` implementou uma SVM linear. A principal motivação foi comparar um modelo baseado em vizinhança com um modelo que aprende uma fronteira linear global.

Aqui, a decisão mais importante foi manter o mesmo protocolo do kNN. A SVM usou:

- os mesmos arquivos preparados;
- as mesmas features;
- o mesmo split development/holdout;
- o mesmo `StratifiedKFold` com 5 folds;
- seleção por `accuracy`;
- avaliação final única no holdout.

O hiperparâmetro central foi `C`, que controla a regularização. Valores menores favorecem uma margem mais regularizada; valores maiores tornam o modelo mais flexível.

O melhor SVM selecionado foi:

- `C = 0.03`
- média de acurácia em validação cruzada: `0.84459`

No holdout, a SVM obteve:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.84880 |
| Balanced accuracy | 0.75871 |
| Precision positiva | 0.73360 |
| Recall/Sensitivity positiva | 0.58482 |
| Specificity | 0.93259 |
| F1 positivo | 0.65082 |
| ROC AUC | 0.89723 |

A SVM teve a melhor **precision positiva** e a melhor **specificity**. Em termos práticos, isso significa que ela foi mais conservadora ao prever `>50K`: quando dizia que uma pessoa estava na classe positiva, tendia a errar menos, mas também deixava passar mais casos positivos do que o kNN.

Esse comportamento aparece na matriz de confusão: a SVM teve menos falsos positivos que o kNN, mas mais falsos negativos. Portanto, ela é útil quando o custo de marcar indevidamente alguém como `>50K` é maior do que o custo de perder alguns positivos.

## 4. Regressão Logística

A regressão logística foi implementada no notebook `notebooks/05_logistic_regression_classifier.ipynb`, seguindo exatamente a mesma base metodológica dos dois modelos anteriores.

Ela entrou no projeto como terceiro classificador por dois motivos:

- é um modelo clássico, interpretável e fortemente ligado ao conteúdo teórico da disciplina;
- permite comparar kNN e SVM com outro modelo linear, mas agora baseado em probabilidades de classe.

O notebook também recebeu uma seção conceitual explicando por que, no scikit-learn, não aparece o cálculo manual dos coeficientes beta. A explicação conecta a formulação da disciplina com a implementação prática: os coeficientes continuam existindo, são estimados numericamente por um solver e podem ser inspecionados em `model.coef_` e `model.intercept_`.

O melhor modelo de regressão logística selecionado foi:

- `C = 0.3`
- média de acurácia em validação cruzada: `0.84386`

No holdout, a regressão logística obteve:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.85049 |
| Balanced accuracy | 0.76396 |
| Precision positiva | 0.73297 |
| Recall/Sensitivity positiva | 0.59694 |
| Specificity | 0.93097 |
| F1 positivo | 0.65800 |
| ROC AUC | 0.89756 |

O principal resultado da regressão logística foi ter alcançado a melhor **accuracy** geral no holdout. Ela ficou ligeiramente acima da SVM e do kNN nesse critério.

Ao mesmo tempo, seu desempenho nas métricas da classe positiva ficou entre SVM e kNN: melhor recall que a SVM, mas menor que o kNN; precision próxima da SVM, mas maior que a do kNN. Isso torna a regressão logística um modelo equilibrado e fácil de explicar, especialmente quando a métrica principal é acurácia geral.

## Comparação final dos três modelos

A comparação consolidada está salva em:

- `submissions/knn_vs_svm_vs_logreg_holdout_comparison.csv`
- `submissions/knn_vs_svm_vs_logreg_holdout_metrics.png`

![Comparação de métricas no holdout](submissions/knn_vs_svm_vs_logreg_holdout_metrics.png)

| Modelo | Accuracy | Balanced accuracy | Precision positiva | Recall/Sensitivity | Specificity | F1 positivo | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| kNN | 0.84819 | 0.77354 | 0.70803 | 0.62946 | 0.91761 | 0.66644 | 0.90215 |
| Linear SVM | 0.84880 | 0.75871 | 0.73360 | 0.58482 | 0.93259 | 0.65082 | 0.89723 |
| Logistic Regression | 0.85049 | 0.76396 | 0.73297 | 0.59694 | 0.93097 | 0.65800 | 0.89756 |

Resumo da comparação:

- **Melhor accuracy:** Regressão Logística.
- **Melhor balanced accuracy:** kNN.
- **Melhor precision positiva:** SVM linear.
- **Melhor recall/sensitivity positiva:** kNN.
- **Melhor specificity:** SVM linear.
- **Melhor F1 positivo:** kNN.
- **Melhor ROC AUC:** kNN.

Assim, não existe uma resposta única se a pergunta for simplesmente "qual modelo é melhor?". A escolha depende do critério.

Se o objetivo for maximizar acurácia geral, a **Regressão Logística** foi a melhor no holdout. Se o objetivo for capturar melhor a classe positiva `>50K`, o **kNN** foi mais forte. Se o objetivo for evitar falsos positivos, a **SVM linear** teve o comportamento mais conservador.

## Conclusão

O maior ganho do projeto não veio de trocar modelos aleatoriamente, mas de consolidar uma metodologia comparável.

A preparação dos dados reduziu ruído e criou uma interface consistente. O kNN estabeleceu o primeiro baseline completo. A SVM adicionou uma fronteira linear regularizada e mostrou um perfil mais conservador. A regressão logística fechou o trio com um modelo probabilístico, interpretável e alinhado à teoria da disciplina.

No protocolo local de holdout, a conclusão principal é:

> A Regressão Logística apresentou a maior acurácia geral, enquanto o kNN apresentou o melhor desempenho nas métricas mais sensíveis à classe positiva. A SVM teve o comportamento mais conservador, com maior precisão positiva e maior especificidade.

Para um relatório acadêmico, essa é a leitura mais sólida: os três modelos foram comparados sob as mesmas condições, e cada um mostrou uma força diferente.
