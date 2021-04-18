from typing import List, Optional

from haystack import Document
from haystack.reader.base import BaseReader

class SimpleReader(BaseReader):
    """
    Simple Reader
    """

    def __init__(
        self,
        top_k: int = 10,
        top_k_per_candidate: int = 4,
        return_no_answers: bool = True
    ):
        """
        Convert the es search results to answers.
        :param top_k: The maximum number of answers to return
        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
        :param return_no_answers: If True, the HuggingFace Transformers model could return a "no_answer" (i.e. when there is an unanswerable question)
        """
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.return_no_answers = return_no_answers

        # TODO context_window_size behaviour different from behavior in FARMReader

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a query in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) probability.
        Example:

         ```python
            |{
            |    'query': 'Who is the father of Arya Stark?',
            |    'answers':[
            |                 {'answer': 'Eddard,',
            |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
            |                 'offset_answer_start': 147,
            |                 'offset_answer_end': 154,
            |                 'probability': 0.9787139466668613,
            |                 'score': None,
            |                 'document_id': '1337'
            |                 },...
            |              ]
            |}
         ```
        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers

        """
        if top_k is None:
            top_k = self.top_k
        # get top-answers for each candidate passage
        answers = []

        for doc in documents:
            answers.append({
                "answer": doc.meta.get('answer'),
                "context": doc.text,
                "offset_start": 0,
                "offset_end": 0,
                "probability": doc.probability,
                "score": doc.score,
                "document_id": doc.id,
                "meta": doc.meta
            })

        # sort answers by their `probability` and select top-k
        answers = sorted(
            answers, key=lambda k: k["probability"], reverse=True
        )
        answers = answers[:top_k]

        results = {"query": query,
                   "answers": answers}

        print(f"Answers : {answers}")
        return results

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None,  batch_size: Optional[int] = None):

        raise NotImplementedError("Batch prediction not yet available in TransformersReader.")
