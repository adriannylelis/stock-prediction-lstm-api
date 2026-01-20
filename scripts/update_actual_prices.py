#!/usr/bin/env python3
"""
Script para atualizar preços reais de predições pendentes no Firestore.

Este script busca todas as predições que ainda não têm o campo 'actual_price'
preenchido, baixa os preços reais via yfinance e atualiza o Firestore.

Útil para rodar periodicamente (ex: via Cloud Scheduler) para comparar
predições com valores reais e calcular métricas de acurácia.

Uso:
    python scripts/update_actual_prices.py [--ticker TICKER] [--limit LIMIT]

Exemplos:
    # Atualizar todas as predições pendentes
    python scripts/update_actual_prices.py

    # Atualizar apenas PETR4.SA
    python scripts/update_actual_prices.py --ticker PETR4.SA

    # Atualizar no máximo 50 predições
    python scripts/update_actual_prices.py --limit 50

Requisitos:
    - google-cloud-firestore
    - yfinance
    - GOOGLE_CLOUD_PROJECT configurado (ou FIRESTORE_EMULATOR_HOST para local)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import yfinance as yf

# Adicionar src/ ao path para importar FirestoreService
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.services.firestore_service import FirestoreService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_actual_price(ticker: str, date: str) -> Optional[float]:
    """
    Busca o preço de fechamento real de uma ação em uma data específica.
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA)
        date: Data no formato YYYY-MM-DD
    
    Returns:
        Preço de fechamento ou None se não encontrado
    """
    try:
        # Converter string para datetime
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Verificar se a data já passou
        if target_date.date() > datetime.now().date():
            logger.debug(f"Data futura {date} - pulando")
            return None
        
        # Baixar dados do yfinance
        # Buscar 5 dias para garantir que pegamos o preço (considera finais de semana)
        start_date = (target_date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"Nenhum dado encontrado para {ticker} em {date}")
            return None
        
        # Tentar encontrar o preço exato da data
        hist.index = hist.index.tz_localize(None)  # Remove timezone
        
        # Buscar preço da data ou data mais próxima
        target_datetime = target_date
        if target_datetime in hist.index:
            price = hist.loc[target_datetime, 'Close']
            logger.debug(f"{ticker} em {date}: R$ {price:.2f}")
            return float(price)
        
        # Se não encontrou data exata, pegar a mais próxima (próximo dia útil)
        future_dates = hist.index[hist.index >= target_datetime]
        if len(future_dates) > 0:
            nearest_date = future_dates[0]
            price = hist.loc[nearest_date, 'Close']
            logger.info(f"{ticker}: Usando preço de {nearest_date.date()} para {date}: R$ {price:.2f}")
            return float(price)
        
        logger.warning(f"Não foi possível encontrar preço para {ticker} em {date}")
        return None
        
    except Exception as e:
        logger.error(f"Erro ao buscar preço de {ticker} em {date}: {str(e)}")
        return None


def update_predictions_for_ticker(
    firestore_svc: FirestoreService,
    ticker: str,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Atualiza predições pendentes para um ticker específico.
    
    Args:
        firestore_svc: Instância do FirestoreService
        ticker: Símbolo da ação
        limit: Número máximo de predições a processar
        dry_run: Se True, apenas simula sem salvar
    
    Returns:
        Dicionário com contadores de sucesso e falha
    """
    logger.info(f"Processando predições pendentes para {ticker}...")
    
    # Buscar predições pendentes
    pending = firestore_svc.get_pending_predictions(ticker)
    
    if not pending:
        logger.info(f"Nenhuma predição pendente encontrada para {ticker}")
        return {"success": 0, "failed": 0, "skipped": 0}
    
    logger.info(f"Encontradas {len(pending)} predições pendentes")
    
    # Limitar se necessário
    if limit:
        pending = pending[:limit]
        logger.info(f"Limitando a {limit} predições")
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for prediction in pending:
        pred_id = prediction['id']
        pred_date = prediction['prediction_date']
        
        logger.info(f"Processando predição {pred_id} ({ticker} @ {pred_date})...")
        
        # Buscar preço real
        actual_price = get_actual_price(ticker, pred_date)
        
        if actual_price is None:
            stats["skipped"] += 1
            continue
        
        # Atualizar Firestore
        if not dry_run:
            try:
                firestore_svc.update_actual_price(ticker, pred_date, actual_price)
                logger.info(f"✅ Atualizado: {ticker} @ {pred_date} = R$ {actual_price:.2f}")
                stats["success"] += 1
            except Exception as e:
                logger.error(f"❌ Erro ao atualizar {pred_id}: {str(e)}")
                stats["failed"] += 1
        else:
            logger.info(f"[DRY RUN] Atualizaria: {ticker} @ {pred_date} = R$ {actual_price:.2f}")
            stats["success"] += 1
    
    return stats


def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description='Atualiza preços reais de predições pendentes no Firestore'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        help='Ticker específico para atualizar (ex: PETR4.SA). Se omitido, processa todos.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Número máximo de predições a processar'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simula a execução sem atualizar o Firestore'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Ativa logs detalhados'
    )
    
    args = parser.parse_args()
    
    # Configurar nível de log
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Banner
    print("=" * 60)
    print("  📊 Update Actual Prices - Firestore")
    print("=" * 60)
    if args.dry_run:
        print("  ⚠️  Modo DRY RUN - Nenhuma alteração será feita")
    print("=" * 60)
    print()
    
    # Inicializar Firestore
    logger.info("Conectando ao Firestore...")
    firestore_svc = FirestoreService()
    
    if not firestore_svc.is_available():
        logger.error("Firestore não está disponível")
        logger.error("Verifique GOOGLE_CLOUD_PROJECT ou FIRESTORE_EMULATOR_HOST")
        sys.exit(1)
    
    logger.info("✅ Conectado ao Firestore")
    
    # Processar ticker específico ou todos
    if args.ticker:
        tickers = [args.ticker]
    else:
        # Buscar todos os tickers com predições pendentes
        # (simplificado - busca manualmente por PETR4.SA como exemplo)
        # Em produção, você poderia manter uma lista de tickers ativos
        logger.info("Nenhum ticker especificado, usando lista padrão...")
        tickers = ['PETR4.SA']  # Adicione mais tickers conforme necessário
    
    # Processar cada ticker
    total_stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for ticker in tickers:
        try:
            stats = update_predictions_for_ticker(
                firestore_svc,
                ticker,
                limit=args.limit,
                dry_run=args.dry_run
            )
            
            # Acumular estatísticas
            for key in total_stats:
                total_stats[key] += stats[key]
                
        except Exception as e:
            logger.error(f"Erro ao processar {ticker}: {str(e)}", exc_info=True)
    
    # Resumo final
    print()
    print("=" * 60)
    print("  📈 Resumo da Execução")
    print("=" * 60)
    print(f"  ✅ Atualizadas com sucesso: {total_stats['success']}")
    print(f"  ⏭️  Puladas (data futura):   {total_stats['skipped']}")
    print(f"  ❌ Falhas:                   {total_stats['failed']}")
    print("=" * 60)
    print()
    
    if args.dry_run:
        logger.info("DRY RUN concluído - nenhuma alteração foi feita")
    else:
        logger.info("Atualização concluída!")
    
    # Exit code baseado em sucesso
    if total_stats['failed'] > 0:
        sys.exit(1)
    elif total_stats['success'] == 0 and total_stats['skipped'] == 0:
        logger.warning("Nenhuma predição foi processada")
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
