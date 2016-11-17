import pytest
import urllib.request
import os
import hashlib
from elasticsearch import Elasticsearch, ConnectionError, RequestError, NotFoundError
from time import sleep

from image_match.elasticsearch_driver import SignatureES
from PIL import Image

test_img_url1 = 'https://camo.githubusercontent.com/810bdde0a88bc3f8ce70c5d85d8537c37f707abe/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f652f65632f4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a70672f36383770782d4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a7067'
test_img_url2 = 'https://camo.githubusercontent.com/826e23bc3eca041110a5af467671b012606aa406/68747470733a2f2f63322e737461746963666c69636b722e636f6d2f382f373135382f363831343434343939315f303864383264653537655f7a2e6a7067'
urllib.request.urlretrieve(test_img_url1, 'test1.jpg')
urllib.request.urlretrieve(test_img_url2, 'test2.jpg')

INDEX_NAME = 'test_environment_{}'.format(hashlib.md5(os.urandom(128)).hexdigest()[:12])
@pytest.fixture(scope='module', autouse=True)
def index_name():
    return INDEX_NAME

@pytest.fixture(scope='function', autouse=True)
def setup_index(request, index_name):
    es = Elasticsearch()
    try:
        es.indices.create(index=index_name)
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
    return SignatureES(es=es, index=index_name)

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


def test_add_image_by_url(ses):
    ses.add_image(test_img_url1)
    ses.add_image(test_img_url2)
    assert True


def test_add_image_by_path(ses):
    ses.add_image('test1.jpg')
    assert True


def test_index_refresh(ses):
    ses.add_image('test1.jpg', refresh_after=True)
    r = ses.search_image('test1.jpg')
    assert len(r) == 1


def test_add_image_as_bytestream(ses):
    with open('test1.jpg', 'rb') as f:
        ses.add_image('bytestream_test', img=f.read(), bytestream=True)
    assert True


def test_add_image_with_different_name(ses):
    ses.add_image('custom_name_test', img='test1.jpg', bytestream=False)
    assert True


def test_lookup_from_url(ses):
    ses.add_image('test1.jpg', refresh_after=True)
    r = ses.search_image(test_img_url1)
    assert len(r) == 1
    assert r[0]['path'] == 'test1.jpg'
    assert 'score' in r[0]
    assert 'dist' in r[0]
    assert 'id' in r[0]


def test_lookup_from_file(ses):
    ses.add_image('test1.jpg', refresh_after=True)
    r = ses.search_image('test1.jpg')
    assert len(r) == 1
    assert r[0]['path'] == 'test1.jpg'
    assert 'score' in r[0]
    assert 'dist' in r[0]
    assert 'id' in r[0]

def test_lookup_from_bytestream(ses):
    ses.add_image('test1.jpg', refresh_after=True)
    with open('test1.jpg', 'rb') as f:
        r = ses.search_image(f.read(), bytestream=True)
    assert len(r) == 1
    assert r[0]['path'] == 'test1.jpg'
    assert 'score' in r[0]
    assert 'dist' in r[0]
    assert 'id' in r[0]

def test_lookup_with_cutoff(ses):
    ses.add_image('test2.jpg', refresh_after=True)
    ses.distance_cutoff=0.01
    r = ses.search_image('test1.jpg')
    assert len(r) == 0


def check_distance_consistency(ses):
    ses.add_image('test1.jpg')
    ses.add_image('test2.jpg', refresh_after=True)
    r = ses.search_image('test1.jpg')
    assert r[0]['dist'] == 0.0
    assert r[-1]['dist'] == 0.42672771706789686


def test_add_image_with_metadata(ses):
    metadata = {'some_info':
                    {'test':
                         'ok!'
                     }
                }
    ses.add_image('test1.jpg', metadata=metadata, refresh_after=True)
    r = ses.search_image('test1.jpg')
    assert r[0]['metadata'] == metadata
    assert 'path' in r[0]
    assert 'score' in r[0]
    assert 'dist' in r[0]
    assert 'id' in r[0]


def test_all_orientations(ses):
    im = Image.open('test1.jpg')
    im.rotate(90, expand=True).save('rotated_test1.jpg')

    ses.add_image('test1.jpg', refresh_after=True)
    r = ses.search_image('rotated_test1.jpg', all_orientations=True)
    assert len(r) == 1
    assert r[0]['path'] == 'test1.jpg'
    assert r[0]['dist'] < 0.05  # some error from rotation

    with open('rotated_test1.jpg', 'rb') as f:
        r = ses.search_image(f.read(), bytestream=True, all_orientations=True)
        assert len(r) == 1
        assert r[0]['dist'] < 0.05  # some error from rotation


def test_duplicate(ses):
    ses.add_image('test1.jpg', refresh_after=True)
    ses.add_image('test1.jpg', refresh_after=True)
    r = ses.search_image('test1.jpg')
    assert len(r) == 2
    assert r[0]['path'] == 'test1.jpg'
    assert 'score' in r[0]
    assert 'dist' in r[0]
    assert 'id' in r[0]


def test_duplicate_removal(ses):
    for i in range(10):
        ses.add_image('test1.jpg')
    sleep(1)
    r = ses.search_image('test1.jpg')
    assert len(r) == 10
    ses.delete_duplicates('test1.jpg')
    sleep(1)
    r = ses.search_image('test1.jpg')
    assert len(r) == 1
