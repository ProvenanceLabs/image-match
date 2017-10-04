from image_match.signature_database_base import SignatureDatabaseBase
from image_match.signature_database_base import normalized_distance
from image_match.signature_database_base import make_record
from datetime import datetime
from itertools import product
from operator import itemgetter
import numpy as np
from collections import deque


class SignatureES(SignatureDatabaseBase):
    """Elasticsearch driver for image-match

    This driver deals with document where all simple words, from 1 to N, are stored
    in a single string field in the document, named "simple_words". The words are
    string separated, like "11111 22222 33333 44444 55555 66666 77777".

    The field is queried with a "match" on "simple_words" with a "minimum_should_match"
    given as a parameter. i.e. the following document:
        {"simple_words": "11111 22222 33333 44444 55555 66666 77777"}
    will be returned by the search_single_record function by the given image words:
        11111 99999 33333 44444 00000 55555 88888
    if minimum_should_match is below or equal to 4, because for words are in common.

    The order of the words in the string field is maintained, although it does not make any
    sort of importance because of the way the field is queried.

    """

    def __init__(self, es, index='images', doc_type='image', timeout='10s', size=100, minimum_should_match=6,
                 *args, **kwargs):
        """Extra setup for Elasticsearch

        Args:
            es (elasticsearch): an instance of the elasticsearch python driver
            index (Optional[string]): a name for the Elasticsearch index (default 'images')
            doc_type (Optional[string]): a name for the document time (default 'image')
            timeout (Optional[int]): how long to wait on an Elasticsearch query, in seconds (default 10)
            size (Optional[int]): maximum number of Elasticsearch results (default 100)
            minimum_should_match (Optional[int]): maximum number of common words in the queried image
                and the document (default 6).
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
        self.minimum_should_match = minimum_should_match

        super(SignatureES, self).__init__(*args, **kwargs)

    def search_single_record(self, rec, pre_filter=None):
        path = rec.pop('path')
        signature = rec.pop('signature')
        if 'metadata' in rec:
            rec.pop('metadata')

        query = {
            'query': {
                'bool': {
                    'must': {
                        'match': {
                            'simple_words': {
                                "query": rec["simple_words"],
                                'minimum_should_match': str(self.minimum_should_match)
                            }
                        },
                    }
                }
            },
            '_source': {'excludes': ['simple_words']}
        }

        if pre_filter is not None:
            query['query']['bool']['filter'] = pre_filter

        # Perform minimum_should_match request
        res = self.es.search(index=self.index,
                             doc_type=self.doc_type,
                             body=query,
                             size=self.size,
                             timeout=self.timeout)['hits']['hits']

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

        self.es.index(index=self.index, doc_type=self.doc_type, body=rec, refresh=refresh_after)

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
                self.es.delete(index=self.index, doc_type=self.doc_type, id=id_tag)

    def add_image(self, path, img=None, bytestream=False, metadata=None, refresh_after=False):
        """Add a single image to the database

        Overwrite the base function to search by flat image (call to make_record with flat=True)

        Args:
            path (string): path or identifier for image. If img=None, then path is assumed to be
                a URL or filesystem path
            img (Optional[string]): usually raw image data. In this case, path will still be stored, but
                a signature will be generated from data in img. If bytestream is False, but img is
                not None, then img is assumed to be the URL or filesystem path. Thus, you can store
                image records with a different 'path' than the actual image location (default None)
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image? If img is None but, this
                argument will be ignored.  If img is not None, and bytestream is False, then the behavior
                is as described in the explanation for the img argument
                (default False)
            metadata (Optional): any other information you want to include, can be nested (default None)

        """
        rec = make_record(path, self.gis, self.k, self.N, img=img, bytestream=bytestream, metadata=metadata, flat=True)
        self.insert_single_record(rec, refresh_after=refresh_after)

    def search_image(self, path, all_orientations=False, bytestream=False, pre_filter=None):
        """Search for matches

        Overwrite the base function to search by flat image (call to make_record with flat=True)

        Args:
            path (string): path or image data. If bytestream=False, then path is assumed to be
                a URL or filesystem path. Otherwise, it's assumed to be raw image data
            all_orientations (Optional[boolean]): if True, search for all combinations of mirror
                images, rotations, and color inversions (default False)
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)
            pre_filter (Optional[dict]): filters list before applying the matching algorithm
                (default None)
        Returns:
            a formatted list of dicts representing unique matches, sorted by dist

            For example, if three matches are found:

            [
             {'dist': 0.069116439263706961,
              'id': u'AVM37oZq0osmmAxpPvx7',
              'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg'},
             {'dist': 0.22484320805049718,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'},
             {'dist': 0.42529792112113302,
              'id': u'AVM37p530osmmAxpPvx9',
              'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
            ]

        """
        img = self.gis.preprocess_image(path, bytestream)

        if all_orientations:
            # initialize an iterator of composed transformations
            inversions = [lambda x: x, lambda x: -x]

            mirrors = [lambda x: x, np.fliplr]

            # an ugly solution for function composition
            rotations = [lambda x: x,
                         np.rot90,
                         lambda x: np.rot90(x, 2),
                         lambda x: np.rot90(x, 3)]

            # cartesian product of all possible orientations
            orientations = product(inversions, rotations, mirrors)

        else:
            # otherwise just use the identity transformation
            orientations = [lambda x: x]

        # try for every possible combination of transformations; if all_orientations=False,
        # this will only take one iteration
        result = []

        orientations = set(np.ravel(list(orientations)))
        for transform in orientations:
            # compose all functions and apply on signature
            transformed_img = transform(img)

            # generate the signature
            transformed_record = make_record(transformed_img, self.gis, self.k, self.N, flat=True)

            l = self.search_single_record(transformed_record, pre_filter=pre_filter)
            result.extend(l)

        ids = set()
        unique = []
        for item in result:
            if item['id'] not in ids:
                unique.append(item)
                ids.add(item['id'])

        r = sorted(unique, key=itemgetter('dist'))
        return r
