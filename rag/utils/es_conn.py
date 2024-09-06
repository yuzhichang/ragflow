import re
import json
import time
import os

import elasticsearch
import copy
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search, Index
from rag.settings import doc_store_logger
from rag import settings
from rag.utils import singleton
from api.utils.file_utils import get_project_base_directory
import polars as pl
from rag.utils.data_store_conn import DocStoreConnection, MatchExpr, OrderByExpr, MatchTextExpr, MatchDenseExpr

doc_store_logger.info("Elasticsearch sdk version: "+str(elasticsearch.__version__))


@singleton
class ESConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        for _ in range(10):
            try:
                self.es = Elasticsearch(
                    settings.ES["hosts"].split(","),
                    basic_auth=(settings.ES["username"], settings.ES["password"]) if "username" in settings.ES and "password" in settings.ES else None,
                    verify_certs=False,
                    timeout=600
                )
                if self.es:
                    self.info = self.es.info()
                    doc_store_logger.info("Connect to es.")
                    break
            except Exception as e:
                doc_store_logger.error("Fail to connect to es: " + str(e))
                time.sleep(1)
        if not self.es.ping():
            raise Exception("Can't connect to ES cluster")
        v = self.info.get("version", {"number": "5.6"})
        v = v["number"].split(".")[0]
        if int(v) < 8:
            raise Exception(f"ES version must be greater than or equal to 8, current version: {v}")
        fp_mapping = os.path.join(get_project_base_directory(), "conf", "mapping.json")
        if not os.path.exists(fp_mapping):
            raise Exception(f"Mapping file not found at {fp_mapping}")
        self.mapping = json.load(open(fp_mapping, "r"))

    """
    Database operations
    """
    def health(self) -> dict:
        return dict(self.es.cluster.health())

    """
    Table operations
    """
    def createIdx(self, indexName: str):
        try:
            from elasticsearch.client import IndicesClient
            return IndicesClient(self.es).create(index=indexName,
                                                 settings=self.mapping["settings"],
                                                 mappings=self.mapping["mappings"])
        except Exception as e:
            doc_store_logger.error("ES create index error %s ----%s" % (indexName, str(e)))

    def deleteIdx(self, indexName: str):
        try:
            return self.es.indices.delete(indexName, allow_no_indices=True)
        except Exception as e:
            doc_store_logger.error("ES delete index error %s ----%s" % (indexName, str(e)))
        
    def indexExist(self, indexName: str) -> bool:
        s = Index(indexName, self.es)
        for i in range(3):
            try:
                return s.exists()
            except Exception as e:
                doc_store_logger.error("ES updateByQuery indexExist: " + str(e))
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue
        return False
        
    """
    CRUD operations
    """
    def search(self, selectFields: list[str], condition: dict, matchExprs: list[MatchExpr], orderBy: OrderByExpr, offset: int, limit: int, indexName: str) -> list[dict] | pl.DataFrame:
        """
        Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
        """
        s = Search()
        bqry = None
        for m in matchExprs:
            if isinstance(m, MatchTextExpr):
                bqry = Q("bool",
                            must=Q("query_string", fields=m.fields,
                                type="best_fields", query=m.matching_text,
                                boost=1)
                        )

        assert(bqry is not None)
        if condition:
            for k, v in condition.items():
                if isinstance(v, list):
                    bqry.filter.append(Q("terms", **{k: v}))
                if isinstance(v, str) or isinstance(v, int):
                    bqry.filter.append(Q("term", **{k: v}))
                else:
                    raise Exception("Condition value must be int, str or list[int|str].")

        for m in matchExprs:
            if isinstance(m, MatchDenseExpr):
                assert(bqry is not None)
                s["knn"] = {
                    "field": m.vector_column_name,
                    "k": m.topn,
                    "similarity": 0.1,
                    "num_candidates": m.topn * 2,
                    "query_vector": list(m.embedding_data),
                }
                s["knn"]["filter"] = bqry.to_dict()

        s = s.query(bqry)
        for field in ["content_ltks", "title_ltks"]:
            s = s.highlight(field)

        if orderBy:
            orders = list()
            for field, order in orderBy.fields:
                order = "asc" if order == 0 else "desc"
                orders.append({field: {"order": order, "unmapped_type": "float",
                                 "mode": "avg", "numeric_type": "double"}})
            s = s.sort(*orders)

        s = s.source(selectFields)
        if limit!=0:
            s = s[offset:limit]
        q = s.to_dict()

        for i in range(3):
            try:
                res = self.es.search(index=(indexName),
                                     body=q,
                                     timeout="600s",
                                     # search_type="dfs_query_then_fetch",
                                     track_total_hits=True,
                                     _source=True)
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                return res
            except Exception as e:
                doc_store_logger.error(
                    "ES search exception: " +
                    str(e) +
                    "【Q】：" +
                    str(q))
                if str(e).find("Timeout") > 0:
                    continue
                raise e
        doc_store_logger.error("ES search timeout for 3 times!")
        raise Exception("ES search timeout.")

    def get(self, docId: str, indexName: str) -> dict:
        for i in range(3):
            try:
                res = self.es.get(index=(indexName),
                                     id=docId)
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                return res
            except Exception as e:
                doc_store_logger.error(
                    "ES get exception: " +
                    str(e) +
                    "【Q】：" +
                    docId)
                if str(e).find("Timeout") > 0:
                    continue
                raise e
        doc_store_logger.error("ES search timeout for 3 times!")
        raise Exception("ES search timeout.")

    def upsertBulk(self, documents: list[dict], indexName: str):
        # Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html
        acts = []
        for d in documents:
            d_copy = copy.deepcopy(d)
            meta_id = d_copy["_id"]
            del d_copy["_id"]
            acts.append(
                {"update": {"_id": meta_id, "_index": indexName}, "retry_on_conflict": 100})
            acts.append({"doc": d_copy, "doc_as_upsert": "true"})

        res = []
        for _ in range(100):
            try:
                r = self.es.bulk(index=(indexName), operations=acts,
                                     refresh=False, timeout="600s")
                if re.search(r"False", str(r["errors"]), re.IGNORECASE):
                    return res

                for it in r["items"]:
                    if "error" in it["update"]:
                        res.append(str(it["update"]["_id"]) +
                                   ":" + str(it["update"]["error"]))
                return res
            except Exception as e:
                doc_store_logger.warn("Fail to bulk: " + str(e))
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                self.conn()
        return res

    def update(self, condition: dict, newValue: dict, indexName: str):
        if 'id' not in condition:
            raise Exception("Condition must contain id.")
        doc = copy.deepcopy(condition)
        id = doc['id']
        del doc['id']
        for i in range(3):
            try:
                self.es.update(index=indexName, id=id, doc=doc)
                return True
            except Exception as e:
                doc_store_logger.error(
                    "ES update exception: " + str(e) + " id:" + str(id) +
                    json.dumps(condition, ensure_ascii=False))
                if str(e).find("Timeout") > 0:
                    continue
        return False

    def delete(self, condition: dict, indexName: str):
        filter = dict()
        for k, v in condition.items():
            if isinstance(v, str) or isinstance(v, int):
                filter[k] = v
            else:
                raise Exception("Condition value must be int, str.")
        for _ in range(10):
            try:
                self.es.delete(
                    index=indexName,
                    **filter,
                    refresh=True,
                    doc_type="_doc")
                doc_store_logger.info("Remove %s" % str(filter))
                return True
            except Exception as e:
                doc_store_logger.warn("Fail to delete: " + str(filter) + str(e))
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                if re.search(r"(not_found)", str(e), re.IGNORECASE):
                    return True
                self.conn()

        doc_store_logger.error("Fail to delete: " + str(filter))
        return False


ELASTICSEARCH = ESConnection()
