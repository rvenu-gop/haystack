import json
import logging
from typing import Dict, Union, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from rest_api.controller.search import PIPELINE

import langid
langid.set_languages(['de', 'en'])  # ISO 639-1 codes
router = APIRouter()
logger = logging.getLogger(__name__)

DB_INDEX_AUTOCOMPLETE = "autocomplete"

retriever = PIPELINE.get_node(name="ESRetriever")
document_store = retriever.document_store if retriever else None
document_store._create_document_index(DB_INDEX_AUTOCOMPLETE)
elasticsearch_client = document_store.get_elastic_client()

router = APIRouter()

class Request(BaseModel):
    search: str

def addQuestionToAutocomplete(question: str):
    # todo: if it already exists; we need to increment count;
    body = {
        'text': question,
        'count' : 1
    }
    res = elasticsearch_client.index(index=DB_INDEX_AUTOCOMPLETE, body=body)

@router.get("/autocomplete")
def ask(search: str):
    interim = elasticsearch_client.search(index=DB_INDEX_AUTOCOMPLETE, body=
    {
        '_source':['text'],
        'query':{
            "bool": {
                "must": [{
                    "match": {
                        "text": search
                    }
                },
                    {
                        "exists": {
                            "field": "count"
                        }
                    }]
            }
        },
        'size': 10,
        'sort' :[
                {'count' : {'order' : 'desc' }}
        ]
    })

    resultCount = len(interim['hits']['hits'])
    result = []
    for i in range(resultCount):
        result.append(interim['hits']['hits'][i]['_source']['text'])

    lang, score = langid.classify(search)

    return {
            "results":result,
            "language": lang
        }
