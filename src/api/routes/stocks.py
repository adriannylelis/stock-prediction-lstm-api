import logging

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

stocks_bp = Blueprint("stocks", __name__)

AVAILABLE_STOCKS = [
    {"symbol": "PETR4.SA", "name": "Petrobras PN", "market": "B3"},
    {"symbol": "VALE3.SA", "name": "Vale ON", "market": "B3"},
    {"symbol": "ITUB4.SA", "name": "Itaú Unibanco PN", "market": "B3"},
    {"symbol": "BBDC4.SA", "name": "Bradesco PN", "market": "B3"},
    {"symbol": "ABEV3.SA", "name": "Ambev ON", "market": "B3"},
    {"symbol": "BBAS3.SA", "name": "Banco do Brasil ON", "market": "B3"},
    {"symbol": "WEGE3.SA", "name": "WEG ON", "market": "B3"},
    {"symbol": "RENT3.SA", "name": "Localiza ON", "market": "B3"},
    {"symbol": "SUZB3.SA", "name": "Suzano ON", "market": "B3"},
    {"symbol": "RAIL3.SA", "name": "Rumo ON", "market": "B3"},
    {"symbol": "JBSS3.SA", "name": "JBS ON", "market": "B3"},
    {"symbol": "MGLU3.SA", "name": "Magazine Luiza ON", "market": "B3"},
    {"symbol": "B3SA3.SA", "name": "B3 ON", "market": "B3"},
    {"symbol": "VIVT3.SA", "name": "Telefônica Brasil ON", "market": "B3"},
    {"symbol": "ELET3.SA", "name": "Eletrobras ON", "market": "B3"},
    {"symbol": "CSNA3.SA", "name": "CSN ON", "market": "B3"},
    {"symbol": "USIM5.SA", "name": "Usiminas PNA", "market": "B3"},
    {"symbol": "GOAU4.SA", "name": "Metalúrgica Gerdau PN", "market": "B3"},
    {"symbol": "CIEL3.SA", "name": "Cielo ON", "market": "B3"},
    {"symbol": "GGBR4.SA", "name": "Gerdau PN", "market": "B3"},
    {"symbol": "EMBR3.SA", "name": "Embraer ON", "market": "B3"},
    {"symbol": "TOTS3.SA", "name": "TOTVS ON", "market": "B3"},
    {"symbol": "RADL3.SA", "name": "Raia Drogasil ON", "market": "B3"},
    {"symbol": "LREN3.SA", "name": "Lojas Renner ON", "market": "B3"},
    {"symbol": "COGN3.SA", "name": "Cogna Educação ON", "market": "B3"},
    {"symbol": "AAPL", "name": "Apple Inc.", "market": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "market": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)", "market": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "market": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "market": "NASDAQ"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "market": "NASDAQ"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "market": "NASDAQ"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "market": "NASDAQ"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "market": "NASDAQ"},
    {"symbol": "INTC", "name": "Intel Corporation", "market": "NASDAQ"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "market": "NYSE"},
    {"symbol": "BAC", "name": "Bank of America Corp.", "market": "NYSE"},
    {"symbol": "WMT", "name": "Walmart Inc.", "market": "NYSE"},
    {"symbol": "V", "name": "Visa Inc.", "market": "NYSE"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "market": "NYSE"},
    {"symbol": "PG", "name": "Procter & Gamble Co.", "market": "NYSE"},
    {"symbol": "DIS", "name": "The Walt Disney Company", "market": "NYSE"},
    {"symbol": "KO", "name": "The Coca-Cola Company", "market": "NYSE"},
    {"symbol": "PFE", "name": "Pfizer Inc.", "market": "NYSE"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "market": "NYSE"},
]


@stocks_bp.route("/stocks", methods=["GET"])
def get_stocks():
    """
    GET /stocks

    Retorna lista de ações disponíveis para previsão.

    Query Parameters:
        market (optional): Filtrar por mercado ('B3', 'NASDAQ', 'NYSE')

    Response 200 OK:
        {
            "success": true,
            "data": [
                {
                    "symbol": "PETR4.SA",
                    "name": "Petrobras PN",
                    "market": "B3"
                },
                ...
            ],
            "count": 45
        }
    """
    try:
        logger.info(f"Retornando {len(AVAILABLE_STOCKS)} ações disponíveis")

        return (
            jsonify(
                {
                    "success": True,
                    "data": AVAILABLE_STOCKS,
                    "count": len(AVAILABLE_STOCKS),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Erro ao buscar lista de ações: {e}")
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "Erro ao buscar lista de ações",
                    "status": 500,
                }
            ),
            500,
        )
