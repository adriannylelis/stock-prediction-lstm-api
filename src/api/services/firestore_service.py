"""
Serviço para gerenciar histórico de predições no Firestore.

Suporta:
- Produção: Usa credenciais do Cloud Run automaticamente
- Dev local: Usa emulador Firestore (FIRESTORE_EMULATOR_HOST)
- Testes: Usa emulador Firestore
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)


class FirestoreService:
    """
    Serviço para persistir e recuperar histórico de predições.

    Schema da collection 'predictions':
    {
        id: "auto-generated",
        ticker: "PETR4.SA",
        predicted_at: "2026-01-19T15:30:00Z",
        prediction_date: "2026-01-20",
        predicted_price: 38.45,
        current_price: 37.85,
        actual_price: null | 38.72,
        error: null | 0.27,
        error_percent: null | 0.70,
        model_version: "v1.0.5"
    }
    """

    def __init__(self):
        """
        Inicializa cliente Firestore.

        Detecta automaticamente o ambiente:
        - Se FIRESTORE_EMULATOR_HOST está definido: usa emulador (dev/test)
        - Caso contrário: usa Firestore em produção (Cloud Run)
        """
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "stock-ml-prod")
        emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST")

        try:
            if emulator_host:
                logger.info(f"🧪 Usando Firestore Emulator: {emulator_host}")
                self.db = firestore.Client(project=project_id)
            else:
                logger.info(f"☁️ Usando Firestore em produção: {project_id}")
                # Cloud Run auto-detecta credenciais via service account
                self.db = firestore.Client(project=project_id)

            self.collection = self.db.collection("predictions")
            self._initialized = True
            logger.info("✅ FirestoreService inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar FirestoreService: {e}")
            self._initialized = False
            raise

    def is_available(self) -> bool:
        """
        Verifica se Firestore está disponível e funcionando.

        Returns:
            bool: True se consegue fazer queries, False caso contrário
        """
        if not self._initialized:
            return False

        try:
            # Tenta fazer uma query simples para validar conexão
            list(self.collection.limit(1).stream())
            return True
        except Exception as e:
            logger.warning(f"⚠️ Firestore indisponível: {e}")
            return False

    def save_prediction(self, data: dict) -> Optional[str]:
        """
        Salva ou atualiza uma predição no Firestore (UPSERT).

        Se já existe uma predição para o mesmo ticker e prediction_date,
        atualiza ao invés de criar duplicata.

        Args:
            data: Dicionário com dados da predição
                - ticker (str): Símbolo da ação
                - prediction_date (str): Data alvo da predição (YYYY-MM-DD)
                - predicted_price (float): Preço previsto
                - current_price (float): Preço atual no momento da predição
                - actual_price (float|None): Preço real (preenchido depois)
                - model_version (str): Versão do modelo

        Returns:
            str: ID do documento criado/atualizado, ou None se falhar
        """
        try:
            ticker = data["ticker"]
            prediction_date = data["prediction_date"]

            # Verificar se já existe predição para esse ticker e data
            existing = (
                self.collection.where("ticker", "==", ticker)
                .where("prediction_date", "==", prediction_date)
                .limit(1)
                .stream()
            )

            existing_doc = None
            for doc in existing:
                existing_doc = doc
                break

            # Adicionar timestamp se não fornecido
            if "predicted_at" not in data:
                data["predicted_at"] = datetime.utcnow().isoformat()

            # Garantir que campos opcionais existam
            data.setdefault("actual_price", None)
            data.setdefault("error", None)
            data.setdefault("error_percent", None)

            if existing_doc:
                # Atualizar predição existente
                doc_id = existing_doc.id
                self.collection.document(doc_id).update(data)
                logger.info(
                    f"🔄 Predição atualizada: {doc_id} | {ticker} | {prediction_date}"
                )
            else:
                # Criar nova predição
                _, doc_ref = self.collection.add(data)
                doc_id = doc_ref.id
                logger.info(
                    f"✅ Predição salva: {doc_id} | {ticker} | {prediction_date}"
                )

            return doc_id

        except Exception as e:
            logger.error(f"❌ Erro ao salvar predição: {e}")
            return None

    def get_predictions(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Retorna histórico de predições para um ticker específico.

        Args:
            ticker: Símbolo da ação (ex: PETR4.SA)
            limit: Número máximo de predições a retornar

        Returns:
            Lista de dicionários com dados das predições (ordenado por data desc)
        """
        try:
            query = (
                self.collection.where(filter=FieldFilter("ticker", "==", ticker))
                .order_by("predicted_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )

            predictions = []
            for doc in query.stream():
                data = doc.to_dict()
                data["id"] = doc.id
                predictions.append(data)

            logger.info(f"📊 {len(predictions)} predições encontradas para {ticker}")
            return predictions

        except Exception as e:
            logger.error(f"❌ Erro ao buscar predições para {ticker}: {e}")
            return []

    def get_pending_predictions(self, ticker: Optional[str] = None) -> List[Dict]:
        """
        Retorna predições que ainda não têm preço real (actual_price == null).

        Args:
            ticker: Opcional - filtrar por ticker específico

        Returns:
            Lista de predições pendentes
        """
        try:
            query = self.collection.where(
                filter=FieldFilter("actual_price", "==", None)
            )

            if ticker:
                query = query.where(filter=FieldFilter("ticker", "==", ticker))

            predictions = []
            for doc in query.stream():
                data = doc.to_dict()
                data["id"] = doc.id
                predictions.append(data)

            logger.info(
                f"📋 {len(predictions)} predições pendentes"
                + (f" para {ticker}" if ticker else "")
            )
            return predictions

        except Exception as e:
            logger.error(f"❌ Erro ao buscar predições pendentes: {e}")
            return []

    def update_actual_price(
        self, ticker: str, prediction_date: str, actual_price: float
    ) -> int:
        """
        Atualiza o preço real de predições pendentes para uma data específica.

        Calcula automaticamente:
        - error = actual_price - predicted_price
        - error_percent = (error / actual_price) * 100

        Args:
            ticker: Símbolo da ação
            prediction_date: Data da predição no formato YYYY-MM-DD
            actual_price: Preço real observado

        Returns:
            int: Número de documentos atualizados
        """
        try:
            # Buscar predições pendentes para este ticker e data
            query = (
                self.collection.where(filter=FieldFilter("ticker", "==", ticker))
                .where(filter=FieldFilter("prediction_date", "==", prediction_date))
                .where(filter=FieldFilter("actual_price", "==", None))
            )

            docs = list(query.stream())
            updated = 0

            for doc in docs:
                predicted_price = doc.to_dict()["predicted_price"]
                error = actual_price - predicted_price
                error_percent = (error / actual_price) * 100 if actual_price != 0 else 0

                # Atualizar documento
                doc.reference.update(
                    {
                        "actual_price": actual_price,
                        "error": round(error, 4),
                        "error_percent": round(error_percent, 2),
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )
                updated += 1

            if updated > 0:
                logger.info(
                    f"✅ {updated} predições atualizadas para {ticker} em {prediction_date}"
                )
            else:
                logger.debug(
                    f"ℹ️ Nenhuma predição pendente encontrada para {ticker} em {prediction_date}"
                )

            return updated

        except Exception as e:
            logger.error(f"❌ Erro ao atualizar preço real: {e}")
            return 0

    def get_accuracy_metrics(self, ticker: str, limit: int = 100) -> Dict:
        """
        Calcula métricas de acurácia para um ticker.

        Métricas calculadas:
        - total: Total de predições com preço real
        - mae: Mean Absolute Error
        - mape: Mean Absolute Percentage Error
        - rmse: Root Mean Squared Error

        Args:
            ticker: Símbolo da ação
            limit: Número de predições recentes a considerar

        Returns:
            Dicionário com métricas de acurácia
        """
        try:
            # Buscar predições com preço real
            query = (
                self.collection.where(filter=FieldFilter("ticker", "==", ticker))
                .where(filter=FieldFilter("actual_price", "!=", None))
                .order_by("predicted_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )

            predictions = [doc.to_dict() for doc in query.stream()]

            if not predictions:
                return {
                    "ticker": ticker,
                    "total": 0,
                    "mae": None,
                    "mape": None,
                    "rmse": None,
                }

            # Calcular métricas
            errors = [
                abs(p["actual_price"] - p["predicted_price"]) for p in predictions
            ]
            mae = sum(errors) / len(errors)

            mape = sum(
                [
                    abs((p["actual_price"] - p["predicted_price"]) / p["actual_price"])
                    * 100
                    for p in predictions
                    if p["actual_price"] != 0
                ]
            ) / len(predictions)

            rmse = (sum([e**2 for e in errors]) / len(errors)) ** 0.5

            return {
                "ticker": ticker,
                "total": len(predictions),
                "mae": round(mae, 4),
                "mape": round(mape, 2),
                "rmse": round(rmse, 4),
            }

        except Exception as e:
            logger.error(f"❌ Erro ao calcular métricas para {ticker}: {e}")
            return {
                "ticker": ticker,
                "total": 0,
                "mae": None,
                "mape": None,
                "rmse": None,
                "error": str(e),
            }
