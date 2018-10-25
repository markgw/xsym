================================
  Codebase for SCiL 2019 paper
================================

 | *Unsupervised Learning of Cross-Lingual Symbol Embeddings Without Parallel Data*
 | Mark Granroth-Wilding and Hannu Toivonen (2019)
 | In proceedings `Society for Computation in Linguistics (SCiL) <https://blogs.umass.edu/scil/scil-2019/>`_

This codebase contains the code used to prepare data and train models for this paper.
In the paper, the model is called **Xsym**. In this code, it is called **neural_sixgram**.

For more information about the paper, including downloadable pre-trained embeddings,
`see here <https://mark.granroth-wilding.co.uk/papers/unsup_symbol/>`_.

It uses `Pimlico <http://pimlico.readthedocs.io>`_.
Pimlico pipeline config files can be found in the ``pipelines`` directory.
Most of the code consists of :doc:`Pimlico modules (documented here) <modules/langsim.modules>`.

The code has been cleaned up for release, which involved removing a lot of
old code from various experiments carried out over a number of years. Hopefully,
I've not removed anything important, but `get in touch with Mark <https://mark.granroth-wilding.co.uk/>`_
if something seems to be missing.


Getting started
===============
To start using the code, see `Pimlico's guide for initializing Pimlico with someone
else's code <https://pimlico.readthedocs.io/en/latest/guides/bootstrap.html>`_.

In short...

 * Download this codebase and extract it.
 * Download the `bootstrap.py <https://raw.githubusercontent.com/markgw/pimlico/master/admin/bootstrap.py>`_ script from Pimlico
   to the root directory of the codebase.
 * In the root directory, run: ``python bootstrap.py pipelines/char_embed_corpora.conf``
 * Check that the setup has worked:

   * ``cd pipelines``
   * ``./char_embed_corpora.conf status``
   * Pimlico should do some initial setup and then show a long list of modules in the pipeline

 * Delete ``bootstrap.py``


Pipelines
=========
There are two pipelines. These cover the corruption experiment and the main model training
described in the paper.

In addition to this, if you want to reproduce everything we did,
you'll need to preprocess the data for low-resourced Uralic languages to clean it up.
That process is implemented and documented in
`a separate codebase <https://mark.granroth-wilding.co.uk/papers/unsup_symbol/>`_,
which also uses Pimlico.

char_embed_corpora
------------------

Main model training pipeline.

This pipeline loads a variety of corpora and trains Xsym on them. It produces all the
models described in the paper. To train on these corpora, you'll need to download
them and then update the paths in the ``[vars]`` section to point to their locations.

There are two slightly different implementations of the training code, found in the
Pimlico modules :mod:`~langsim.modules.local_lm.neural_sixgram`
and :mod:`~langsim.modules.local_lm.neural_sixgram2`. If you're training the model
yourself, you should use the more recent and more efficient :mod:`~langsim.modules.local_lm.neural_sixgram2`.

The pipeline also includes training on some language pairs not reported in the paper.

char_embed_corrupt
------------------

Language corruption experiments to test Xsym's robustness to different types of noise.

This pipeline implements the language corruption experiments reported in the paper.
It takes real language data (Finnish forum posts) and applies random corruptions to
it, training Xsym on uncorrupted and corrupted pairs.


Corpora
=======

Ylilauta
--------
Finnish forum posts.

 * `Download <http://urn.fi/urn:nbn:fi:lb-2016101211>`_.

Estonian Reference Corpus
-------------------------
Corpus of written Estonian from a variety of sources.
Here we use just the subsets: ``tasakaalus_ajalehed`` and ``foorumid_lausestatud``.

 * `Corpus <http://www.cl.ut.ee/korpused/segakorpus/>`_
 * `Forum post subset <http://www.cl.ut.ee/korpused/segakorpus/uusmeedia/foorumid>`_

Danish Wikipedia dump
---------------------
Text dump of Danish Wikipedia.

 * `Download <http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/>`_

Europarl
--------
The Europarl corpus of transcripts from the European Parliament.

Download the full source release. We use the Swedish, Spanish and Portuguese parts.

 * `Homepage <http://www.statmt.org/europarl/>`_

Multilingual Resource Collection of the University of Helsinki Language Corpus Server (UHLCS)
---------------------------------------------------------------------------------------------
Data used to be available from the homepage, but is now available through the CSC.
You'll need to request access to the specific language datasets used.

The data you get is messy, in inconsistent formats and encodings. See the code distributed
separately for how to preprocess this and get it into a useable textual form, which we use below.

 * `UHLCS homepage <http://www.ling.helsinki.fi/uhlcs/>`_
 * `CSC <https://www.csc.fi/>`_
 * `My code for preparing the corpora <https://mark.granroth-wilding.co.uk/papers/unsup_symbol/>`_


Documentation
=============

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Pimlico modules <modules/langsim.modules>
   API docs <api/langsim>
