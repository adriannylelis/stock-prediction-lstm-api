Stock Prediction LSTM 📈
Este repositório contém a solução para o Tech Challenge - Fase 4 da Pós-Graduação em Machine Learning Engineering. O objetivo é desenvolver um modelo preditivo de Deep Learning utilizando arquitetura LSTM (Long Short-Term Memory) para prever o fechamento de ações da bolsa de valores, produtizando o resultado através de uma API RESTful containerizada.
🗂️ Estrutura do Projeto
O projeto segue uma arquitetura de monólito modular para facilitar a colaboração e o compartilhamento de artefatos entre as etapas de modelagem e engenharia.

/ 
├── /notebooks          #  Análise exploratória (EDA), testes e gráficos
├── /src
│   ├── /model_training #  Scripts Python para treinar e salvar o modelo
│   └── /api            #  Código da API (main.py, schemas, rotas)
├── /artifacts          # Ponto de encontro: Onde o modelo treinado e o scaler são salvos
│   ├── model.pt        # Modelo serializado
│   └── scaler.pkl      # Objeto Scaler para normalização/desnormalização
├── Dockerfile           #  Configuração da imagem Docker para a API
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação principal

🛠️ Tecnologias Utilizadas
• Linguagem: Python
• Coleta de Dados: yfinance (Yahoo Finance)
• Modelagem: PyTorch (LSTM)
• API:Flask
• Containerização: Docker
