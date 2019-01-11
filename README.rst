Feature Reduction for Network Traffic Analysis
==============================================

This repository contains code for experiments using multiple techniques for reducing network traffic features vectors.
The code was used in *(paper under review)*.

The code here contains all that is necessary to run all the experiments in the paper, as well as the proposed
Traffic Flow Mapper prototype.

Running
-------

First run ``get_data.sh`` to download the data.

Then run ``run.py``:

.. code:: bash

    python run.py -h

By asking for help for each method, you'll get a list of the parameters and default values.
For example, for the ``cats_ae`` method:

.. code:: bash

    python cats_ae -h

As an example, running ``cats_ae``, with ``$DATAPATH`` the path where the data is:

.. code:: bash

    python run.py --size 2 --number 5 cats_ae --reconstruct_loss mse --reconstruct_weight $DATAPATH

Using Docker
~~~~~~~~~~~~

You can use Docker to run this code, without need for downloading the source.
To do that, run:

.. code:: bash

    docker run dcferreira/network_analysis_feature_reduction -h

That command should output the help message.

To run the same example as above, run

.. code:: bash

    docker run dcferreira/network_analysis_feature_reduction --size 2 --number 5 cats_ae --reconstruct_loss mse --reconstruct_weight $DATAPATH


Traffic Flow Mapper
-------------------

This repository contains also a prototype tool for visualizing network traffic flows.

.. image:: https://cn.tuwien.ac.at/assets/projects/tfm/tfm-screen.png

Running with Docker
~~~~~~~~~~~~~~~~~~~

You can easily run the Traffic Flow Mapper prototype with docker:

.. code:: bash

    docker run -p 5006:5006 dcferreira/network_analysis_feature_reduction tfm

To access it, navigate with your browser to http://localhost:5006.
