import unittest
from image_match.elasticsearchflat_driver import SignatureES

from tests.elasticsearch_helper import BaseTestsParent, DOC_TYPE, INDEX_NAME


class ElasticSearchFlatTestSuite(BaseTestsParent.BaseTests):
    @property
    def ses(self):
        """
        Override the ses property to use the flat driver.
        :return: SignatureES from image_match.elasticsearchflat_driver
        """
        es = self.es
        return SignatureES(es=es, index=INDEX_NAME, doc_type=DOC_TYPE)
