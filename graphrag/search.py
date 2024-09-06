#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json

import pandas as pd
from utils.data_store_conn import OrderByExpr

from rag.nlp.search import Dealer


class KGSearch(Dealer):
    def search(self, req, idxnm, emb_mdl=None, highlight=False):
        def merge_into_first(sres, title=""):
            df,texts = [],[]
            for d in sres["hits"]["hits"]:
                try:
                    df.append(json.loads(d["_source"]["content_with_weight"]))
                except Exception:
                    texts.append(d["_source"]["content_with_weight"])
                    pass
            if not df and not texts: return False
            if df:
                try:
                    sres["hits"]["hits"][0]["_source"]["content_with_weight"] = title + "\n" + pd.DataFrame(df).to_csv()
                except Exception:
                    pass
            else:
                sres["hits"]["hits"][0]["_source"]["content_with_weight"] = title + "\n" + "\n".join(texts)
            return True

        src = req.get("fields", ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd",
                                 "image_id", "doc_id", "q_512_vec", "q_768_vec", "position_int", "name_kwd",
                                 "q_1024_vec", "q_1536_vec", "available_int", "content_with_weight",
                                 "weight_int", "weight_flt", "rank_int"
                                 ])

        qst = req.get("question", "")
        matchText, keywords = self.qryr.question(qst, min_match="5%")
        condition = self.get_filters(req)

        ## Entity retrieval
        condition.update({"knowledge_graph_kwd": ["entity"]})
        q_vec = []
        if req.get("vector"):
            assert emb_mdl, "No embedding model selected"
            matchDense = self.get_vector(qst, emb_mdl, 1024)
            q_vec = matchDense.embedding_data

        ent_res = self.dataStore.search(src, condition, [matchText], OrderByExpr(), idxnm)
        if(len(ent_res)>32):
            ent_res = ent_res[0:32]
        entities = [d["name_kwd"] for d in ent_res]
        ent_ids = ent_res["doc_id"]
        if merge_into_first(ent_res, "-Entities-"):
            ent_ids = ent_ids[0:1]

        ## Community retrieval
        condition = self.get_filters(req)
        condition.update({"entities_kwd": entities, "knowledge_graph_kwd": ["community_report"]})
        comm_res = self.dataStore.search(src, condition, [matchText], OrderByExpr(), idxnm)
        if(len(comm_res)>32):
            comm_res = comm_res[0:32]
        comm_ids = comm_res["doc_id"]
        if merge_into_first(comm_res, "-Community Report-"):
            comm_ids = comm_ids[0:1]

        ## Text content retrieval
        condition = self.get_filters(req)
        condition.update({"knowledge_graph_kwd": ["text"]})
        txt_res = self.dataStore.search(src, condition, [matchText], OrderByExpr(), idxnm)
        if(len(txt_res)>6):
            txt_res = txt_res[0:6]
        txt_ids = txt_res["doc_id"]
        if merge_into_first(txt_res, "-Original Content-"):
            txt_ids = txt_ids[0:1]

        return self.SearchResult(
            total=len(ent_ids) + len(comm_ids) + len(txt_ids),
            ids=[*ent_ids, *comm_ids, *txt_ids],
            query_vector=q_vec,
            field={**self.getFields(ent_res, src), **self.getFields(comm_res, src), **self.getFields(txt_res, src)},
            keywords=[]
        )

