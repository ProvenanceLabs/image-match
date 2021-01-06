import pytest
import os
import hashlib
try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve
from elasticsearch import Elasticsearch, ConnectionError, RequestError, NotFoundError
from time import sleep

from image_match.elasticsearch_driver import SignatureES
from PIL import Image

test_img_url1 = 'https://camo.githubusercontent.com/810bdde0a88bc3f8ce70c5d85d8537c37f707abe/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f652f65632f4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a70672f36383770782d4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a7067'
test_img_url2 = 'https://camo.githubusercontent.com/826e23bc3eca041110a5af467671b012606aa406/68747470733a2f2f63322e737461746963666c69636b722e636f6d2f382f373135382f363831343434343939315f303864383264653537655f7a2e6a7067'
urlretrieve(test_img_url1, 'test1.jpg')
urlretrieve(test_img_url2, 'test2.jpg')

INDEX_NAME = 'test_environment_{}'.format(hashlib.md5(os.urandom(128)).hexdigest()[:12])
DOC_TYPE = 'image'
MAPPINGS = {
  "mappings": {
    DOC_TYPE: {
      "dynamic": True,
      "properties": {
        "metadata": {
            "type": "nested",
            "dynamic": True,
            "properties": {
                "tenant_id": { "type": "keyword" },
                "project_id": { "type": "keyword" }
            }
        }
      }
    }
  }
}


@pytest.fixture(scope='module', autouse=True)
def index_name():
    return INDEX_NAME

@pytest.fixture(scope='function', autouse=True)
def setup_index(request, index_name):
    es = Elasticsearch()
    try:
        es.indices.create(index=index_name, body=MAPPINGS)
    except RequestError as e:
        if e.error == u'index_already_exists_exception':
            es.indices.delete(index_name)
        else:
            raise

    def fin():
        try:
            es.indices.delete(index_name)
        except NotFoundError:
            pass

    request.addfinalizer(fin)

@pytest.fixture(scope='function', autouse=True)
def cleanup_index(request, es, index_name):
    def fin():
        try:
            es.indices.delete(index_name)
        except NotFoundError:
            pass
    request.addfinalizer(fin)

@pytest.fixture
def es():
    return Elasticsearch()

@pytest.fixture
def ses(es, index_name):
    return SignatureES(es=es, index=index_name, doc_type=DOC_TYPE)

def test_elasticsearch_running(es):
    i = 0
    while i < 5:
        try:
            es.ping()
            assert True
            return
        except ConnectionError:
            i += 1
            sleep(2)

    pytest.fail('Elasticsearch not running (failed to connect after {} tries)'
                .format(str(i)))


def test_lookup_with_filter_by_metadata(ses):

    ses.add_image('test1.jpg', metadata=_metadata('foo', 'project-x'), refresh_after=True)
    ses.add_image('test2.jpg', metadata=_metadata('foo', 'project-x'), refresh_after=True)
    ses.add_image('test3.jpg', img='test1.jpg', metadata=_metadata('foo', 'project-y'), refresh_after=True)

    ses.add_image('test2.jpg', metadata=_metadata('bar', 'project-x'), refresh_after=True)

    r = ses.search_image('test1.jpg', pre_filter=_nested_filter('foo', 'project-x'))
    assert len(r) == 2

    r = ses.search_image('test1.jpg', pre_filter=_nested_filter('foo', 'project-z'))
    assert len(r) == 0

    r = ses.search_image('test1.jpg', pre_filter=_nested_filter('bar', 'project-x'))
    assert len(r) == 1

    r = ses.search_image('test1.jpg', pre_filter=_nested_filter('bar-2', 'project-x'))
    assert len(r) == 0

    r = ses.search_image('test1.jpg', pre_filter=_nested_filter('bar', 'project-z'))
    assert len(r) == 0

def _metadata(tenant_id, project_id):
    return dict(
            tenant_id=tenant_id,
            project_id=project_id
    )

def _nested_filter(tenant_id, project_id):
    return {
        "nested" : {
            "path" : "metadata",
            "query" : {
                "bool" : {
                    "must" : [
                        {"term": {"metadata.tenant_id": tenant_id}},
                        {"term": {"metadata.project_id": project_id}}
                    ]
                }
             }
        }
    }
