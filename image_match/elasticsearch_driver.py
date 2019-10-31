from .signature_database_base import SignatureDatabaseBase
from .signature_database_base import normalized_distance
from datetime import datetime
import numpy as np
from collections import deque


class SignatureES(SignatureDatabaseBase):
    """Elasticsearch driver for image-match

    """

    def __init__(self, es, index='images', doc_type='image', timeout='10s', size=100,
                 *args, **kwargs):
        """Extra setup for Elasticsearch

        Args:
            es (elasticsearch): an instance of the elasticsearch python driver
            index (Optional[string]): a name for the Elasticsearch index (default 'images')
            doc_type (Optional[string]): a name for the document time (default 'image')
            timeout (Optional[int]): how long to wait on an Elasticsearch query, in seconds (default 10)
            size (Optional[int]): maximum number of Elasticsearch results (default 100)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        Examples:
            >>> from elasticsearch import Elasticsearch
            >>> from image_match.elasticsearch_driver import SignatureES
            >>> es = Elasticsearch()
            >>> ses = SignatureES(es)
            >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            [
             {'dist': 0.0,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.es = es
        self.index = index
        self.doc_type = doc_type
        self.timeout = timeout
        self.size = size

        super(SignatureES, self).__init__(*args, **kwargs)

    def search_single_record(self, rec, pre_filter=None):
        path = rec.pop('path')
        signature = rec.pop('signature')
        if 'metadata' in rec:
            rec.pop('metadata')

        # build the 'should' list
        should = [{'term': {word: rec[word]}} for word in rec]
        body = {
            'query': {
                   'bool': {'should': should}
            },
            '_source': {'excludes': ['simple_word_*']}
        }

        if pre_filter is not None:
            body['query']['bool']['filter'] = pre_filter

        res = self.es.search(
            index=self.index,
            body=body,
            size=self.size,
            timeout=self.timeout
        )['hits']['hits']

        sigs = np.array([x['_source']['signature'] for x in res])

        if sigs.size == 0:
            return []

        dists = normalized_distance(sigs, np.array(signature))

        formatted_res = [{'id': x['_id'],
                          'score': x['_score'],
                          'metadata': x['_source'].get('metadata'),
                          'path': x['_source'].get('url', x['_source'].get('path'))}
                         for x in res]

        for i, row in enumerate(formatted_res):
            row['dist'] = dists[i]
        formatted_res = filter(lambda y: y['dist'] < self.distance_cutoff, formatted_res)

        return formatted_res

    def insert_single_record(self, rec, refresh_after=False):
        rec['timestamp'] = datetime.now()
        self.es.index(
            index=self.index, body=rec, refresh=refresh_after
        )

    def delete_duplicates(self, path):
        """Delete all but one entries in elasticsearch whose `path` value is equivalent to that of path.
        Args:
            path (string): path value to compare to those in the elastic search
        """
        matching_paths = [item['_id'] for item in
                          self.es.search(body={'query':
                                               {'match':
                                                {'path': path}
                                               }
                                              },
                                         index=self.index)['hits']['hits']
                          if item['_source']['path'] == path]
        if len(matching_paths) > 0:
            for id_tag in matching_paths[1:]:
                self.es.delete(index=self.index, id=id_tag)
