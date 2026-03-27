"""
persistence.py
──────────────
Lightweight wrapper around Azure Cosmos DB for any orchestrator that needs to
store user-scoped request data.

Usage
-----
    from persistence import CosmosPersister

    persister = CosmosPersister()          # picks up env-vars automatically
    user_doc  = persister.load_user_doc("alice")
    persister.save_user_doc(user_doc)      # upsert

The module is **framework-agnostic** and contains *no* business logic.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime as dt
from typing import Any, Dict

from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


class CosmosPersister:
    """Tiny DAO for a single Cosmos DB container (partition-key = `/user_id`)."""

    # --------------------------------------------------------------------- #
    # Construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        database_name: str | None = None,
        container_name: str | None = None,
        partition_path: str = "/user_id",
        throughput: int = 400,
        credential=None,
    ):
        endpoint = endpoint or os.environ["COSMOSDB_ENDPOINT"]
        database_name = database_name or os.environ["COSMOSDB_DATABASE_NAME"]
        container_name = container_name or os.environ["COSMOSDB_CONTAINER_MORTGAGE_DATA_NAME"]
        credential = credential or DefaultAzureCredential()

        client = CosmosClient(endpoint, credential)              # type: ignore[arg-type]
        db = client.create_database_if_not_exists(database_name)
        self._container = db.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path=partition_path),
            offer_throughput=throughput,
        )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def load_user_doc(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch the user document.  If it doesn’t exist, create a bare scaffold
        so callers can immediately start appending requests.
        """
        try:
            query = "SELECT * FROM c WHERE c.id = @userId"
            items = list(
                self._container.query_items(
                    query=query,
                    parameters=[{"name": "@userId", "value": user_id}],
                    enable_cross_partition_query=True,
                )
            )

            if items:
                return items[0]

            logger.info("User %s not found – creating empty document.", user_id)
            doc = {
                "id": user_id,
                "user_id": user_id,
                "requests": [],
                "created_utc": dt.utcnow().isoformat(),
                "updated_utc": dt.utcnow().isoformat(),
            }
            self._container.upsert_item(doc, user_id)            # positional PK
            return doc
        except Exception as exc:                                  # pragma: no cover
            logger.error("load_user_doc failed: %s", exc)
            raise

    def save_user_doc(self, doc: Dict[str, Any]) -> None:
        """Upsert the document, bumping its `updated_utc` timestamp."""
        doc["updated_utc"] = dt.utcnow().isoformat()
        self._container.upsert_item(doc, doc["user_id"])

    @staticmethod
    def get_request_ref(user_doc: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        Return a **reference** to the nested request dict so callers can mutate
        it in place.  Raises `ValueError` if not found.
        """
        for req in user_doc.get("requests", []):
            if req.get("request_id") == request_id:
                return req
        raise ValueError(f"Request {request_id} not found for user {user_doc['id']}")
    
   
