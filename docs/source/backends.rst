Other database backends
=======================
Though we designed ``image-match`` with Elasticsearch in mind, other database
backends are possible. For demonstration purposes we include also a `MongoDB`_
driver:

.. code-block:: python

    from image_match.mongodb_driver import SignatureMongo
    from pymongo import MongoClient

    client = MongoClient(connect=False)
    c = client.images.images

    ses = SignatureMongo(c)

now you can use the same functionality as above like ``ses.add_image(...)``.

We tried to separate signature logic from the database insertion/search as much
as possible.  To write your own database backend, you can inherit from the
``SignatureDatabaseBase`` class and override the appropriate methods:

.. code-block:: python

    from image_match.signature_database_base import SignatureDatabaseBase
    # other relevant imports

    class MySignatureBackend(SignatureDatabaseBase):
    
        # if you need to do some setup, override __init__
        def __init__(self, myarg1, myarg2, *args, **kwargs):
            # do some initializing stuff here if necessary
            # ...
            super(MySignatureBakend, self).__init__(*args, **kwargs)
    
        # you MUST implement these two functions
        def search_single_record(self, rec):
            # should query your database given a record generated from
            # signature_database_base.make_record
            # ...
            # should return a list of dicts like 
            # [{'id': 'some_unique_id_from_db',
            #   'dist': 0.109234,
            #   'path': 'url/or/filepath'},
            #  {...}, ...]
            # you can have other keys, but you need at least id and dist
            return formatted_results
    
        def insert_single_record(self, rec):
            # if your database driver or instance can accept a dict as input,
            # this should be very simple
    
        # ...

Unfortunately, implementing a good ``search_single_record`` function does
require some knowledge of `the search algorithm`_. You can also look at the two
included database drivers for guidelines.



.. _MongoDB: https://www.mongodb.org/
.. _the search algorithm: http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps
